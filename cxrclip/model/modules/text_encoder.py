from typing import Optional
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers import BertTokenizerFast
import torch
import os

class HuggingfaceTextEncoder(nn.Module):
    def __init__(
        self,
        name: str = "bert-base-uncased",
        tokenizer = None,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = "~/.cache/huggingface/hub",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        dual_cls: bool = False,
        custom_pretrain_weights = None
    ):
        super().__init__()
        vocab_size = tokenizer.vocab_size

        # using default pretrained weights downloaded from huggingface.
        if pretrained:
            self.text_encoder = AutoModel.from_pretrained(
                cache_dir,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            # intialize the bert model according to the config but no pretrained weights
            model_config = AutoConfig.from_pretrained(
                cache_dir,
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            self.text_encoder = AutoModel.from_config(model_config)
            print('Initialize the text encoder with random weights.')

        if gradient_checkpointing and self.text_encoder.supports_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()

        # load the pretrained weights if asked for.
        if custom_pretrain_weights is not None and pretrained:
            missing, unexpected = self.text_encoder.load_state_dict(custom_pretrain_weights, strict=True)
        self.dual_cls = dual_cls
        if dual_cls:
            self.text_encoder.resize_token_embeddings(vocab_size)
            with torch.no_grad():
                self.text_encoder.get_input_embeddings().weight[tokenizer.cls2_token_id] = \
                    self.text_encoder.get_input_embeddings().weight[tokenizer.cls_token_id].clone()

            # sanity check
            e1 = self.text_encoder.get_input_embeddings().weight[tokenizer.cls_token_id]
            e2 = self.text_encoder.get_input_embeddings().weight[tokenizer.cls2_token_id]
            assert torch.allclose(e1, e2), 'Initial weights for CLS and CLS2 are not the same.'

        self.out_dim = self.text_encoder.config.hidden_size

    def forward(self, x, last_n_hidden_layers=None):
        outputs = self.text_encoder(
            **x,
            output_hidden_states=True,
            return_dict=True
        )
        # retrieve the hidden states of interest
        hidden_states = [
            outputs['hidden_states'][i] for i in last_n_hidden_layers
        ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []
        results = { 
            'last_hidden_state': outputs["last_hidden_state"], # (batch, seq_len, hidden_size), including the CLS and possibly CLS2 token
            'hidden_states': hidden_states # NOTE: this could be the potential cause of out of memory issue
        }
        del outputs # free up memory
        return results

class CustomBertTokenizer(BertTokenizerFast):
    """Enable DUAL CLS token forward pass"""
    
    def __init__(self, cache_dir, dual_cls=False, **kwargs):
        super().__init__(cache_dir=cache_dir, vocab_file=os.path.join(cache_dir, 'vocab.txt'), **kwargs)

        self.dual_cls = dual_cls
        if self.bos_token_id is None:
            self.bos_token_id = self.cls_token_id

        if dual_cls:
            TOKEN = '[CLS2]'

            # Add your custom special token
            special_tokens = {"additional_special_tokens": [TOKEN]}
            num_added = self.add_special_tokens(special_tokens)
            assert num_added >= 1, "The special dual_cls token is not added."

            # Store the custom token ID for easy access
            self.cls2_token_id = self.convert_tokens_to_ids(TOKEN)

            print('Additional [CLS2] token is added to the tokenizer.')

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
        ) -> list[int]:
        """
        Build model inputs with custom token prepended.
        Format: [CLS] + [CUSTOM] + tokens + [SEP]

        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`list[int]`): The first tokenized sequence.
            token_ids_1 (`list[int]`, *optional*): The second tokenized sequence.

        Returns:
            `list[int]`: The model input with special tokens.
        """

        cls_token_id = [self.cls_token_id]
        cls2_token_id = [self.cls2_token_id] if self.dual_cls else [] 
        sep_token_id = [self.sep_token_id] if self.sep_token_id else []
        if token_ids_1 is None:
            return cls_token_id + cls2_token_id + token_ids_0 + sep_token_id

        assert False, "Should not have two sentences"

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`list[int]`): The first tokenized sequence.
            token_ids_1 (`list[int]`, *optional*): The second tokenized sequence.

        Returns:
            `list[int]`: The token type ids.
        """
        cls_offset = 2 if self.dual_cls else 1
        sep_offset = 1 if self.sep_token_id is not None else 0
        if token_ids_1 is None:
            return [0] * (cls_offset + len(token_ids_0) + sep_offset)

        raise ValueError("Custom tokenizer does not support sentence pairs with dual CLS")

    @property
    def vocab_size(self):
        """Override to always return current vocab length"""
        return len(self.get_vocab())
