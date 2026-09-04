from torch import nn
from transformers import AutoModel, AutoConfig
from cxrclip.model.modules.xraydino_utils import XrayDINOLocal
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2Config

class XrayDinov2_224(nn.Module):
    """
    model downloaded from https://huggingface.co/StanfordAIMI/dinov2-base-xray-224
    """
    def __init__(self, ckpt_dir, freeze_backbone=False):
        super().__init__()
        config = AutoConfig.from_pretrained(ckpt_dir, local_files_only=True)
        config.output_hidden_states = True  # need this to access the intermediate layer outputs

        self.model = AutoModel.from_pretrained(
            ckpt_dir,
            config=config,
            dtype="auto",      # or "auto" for GPU
            trust_remote_code=True,
            local_files_only=True
        )
        self.out_dim = 768
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            print('[XrayDINOv2]: Disabling training parameters of the whole xraydinov2 backbone.')
            for p in self.model.parameters():
                p.requires_grad = False
            self.vision_head = Dinov2Encoder(
                config=Dinov2Config(
                    num_hidden_layers=2,
                    hidden_size=768,
                    _attn_implementation="sdpa" # required to manually set.
                )
            )
            print('initiated addtional two dinov2 encoder layer for training (dino.txt).')

    def forward(self, x, last_n_hidden_layers=None):

        if not self.freeze_backbone:
            outputs = self.model(x)
            last_hidden_state = outputs.last_hidden_state
            preserved_hidden_states = [
                outputs.hidden_states[i] for i in last_n_hidden_layers
            ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []
            del outputs # free up some memory
            cls_tokens, patch_tokens = last_hidden_state[:, 0], last_hidden_state[:, 1:]
        else:
            outputs = self.model(x)
            last_hidden_state = outputs.last_hidden_state
            finaloutput = self.vision_head(last_hidden_state)
            cls_tokens, patch_tokens = finaloutput.last_hidden_state[:, 0], finaloutput.last_hidden_state[:, 1:]
            preserved_hidden_states = []
            del outputs # free up some memory

        return { 
            'cls_token': cls_tokens.unsqueeze(1),
            'patch_tokens': patch_tokens,
            'hidden_states': preserved_hidden_states
        }

class XrayDINO(nn.Module):
    
    def __init__(self, ckpt_dir, freeze_backbone, interpolate_pos_encoding=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.xraydino_encoder = XrayDINOLocal(ckpt_dir, freeze_backbone, interpolate_pos_encoding)
        self.out_dim = 768

        # two transformer header on top of the vision backbone
        if self.freeze_backbone:
            self.vision_head = Dinov2Encoder(
                config=Dinov2Config(
                    num_hidden_layers=2,
                    hidden_size=768,
                    _attn_implementation="sdpa" # required to manually set.
                )
            )
            print('initiated addtional two dinov2 encoder layer for training (dino.txt).')

    def forward(self, x, last_n_hidden_layers=None):
        hidden_states = None
        if not self.freeze_backbone:
            cls_tokens, patch_tokens, hidden_states = self.xraydino_encoder.extract_features(x)
            hidden_states = [
                hidden_states[i] for i in last_n_hidden_layers
            ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []
        else:
            # freeze the features
            radino_output = self.xraydino_encoder.model(x, output_hidden_states=True)
            last_hidden_state = radino_output.last_hidden_state
            finaloutput = self.vision_head(last_hidden_state)
            # true last layer output
            cls_tokens, patch_tokens = finaloutput.last_hidden_state[:, 0], finaloutput.last_hidden_state[:, 1:]
            del radino_output # free up some memory
            hidden_states = None

        results = { 
            'cls_token': cls_tokens.unsqueeze(1),
            'patch_tokens': patch_tokens
        } | { 'hidden_states': hidden_states if hidden_states is not None else [] } 
        return results