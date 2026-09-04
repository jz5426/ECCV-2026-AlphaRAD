import logging
from typing import Dict
import torch
from torch import nn
from cxrclip.util.utils import flair_attention_util
from .modules import load_image_encoder, load_projection_head, load_text_encoder

log = logging.getLogger(__name__)

class BaseClip(nn.Module):
    def __init__(self, model_config: Dict, all_loss_config: Dict, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_encoder = load_image_encoder(model_config["image_encoder"])
        self.text_encoder = load_text_encoder(model_config["text_encoder"], tokenizer=tokenizer)
        self.text_pooling = model_config["text_encoder"]["pooling"]
        self.model_config = model_config
        self.loss_config = {k: v for k, v in all_loss_config.items()}
        
        self.vision_dual_cls = self.model_config['image_encoder']['dual_cls']
        self.text_dual_cls = self.model_config['text_encoder']['dual_cls']

        self.projection = "projection_head" in model_config
        self.dual_text_forwarding = True

        self.image_cls_projection, self.text_cls_projection = None, None
        self.image_cls2_projection, self.text_cls2_projection = None, None

        if self.projection:

            # the default cls token will get the projector 
            self.image_cls_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, 
                config_projection_head=model_config["projection_head"]
            )
            self.text_cls_projection = load_projection_head(
                embedding_dim=self.text_encoder.out_dim, 
                config_projection_head=model_config["projection_head"]
            )
        else:
            assert (
                self.image_encoder.out_dim == self.text_encoder.out_dim
            ), "Without 'projection_head', embedding_dim of the image and text encoder must be the same."

    def encode_image(self, image, last_n_hidden_layers=None):
        image_features = self.image_encoder(image, last_n_hidden_layers)

        if self.model_config["image_encoder"]["name"] == "resnet":
            return image_features
        elif 'swin' in self.model_config["image_encoder"]["name"]:
            return image_features[:, 0]
        else:
            # for this project.
            results = { 
                'cls_token': image_features['cls_token'], 
                'patch_tokens': image_features['patch_tokens'],
                'hidden_states': image_features['hidden_states'] if 'hidden_states' in image_features else None
                }
            return results

    def encode_text(self, text_tokens, last_n_hidden_layers=None):
        text_features = self.text_encoder(text_tokens, last_n_hidden_layers) # outputs check text_encoder.py
        assert self.text_pooling in {'bos', 'mean'}, 'Should be using bos or mean pooling.'

        text_patch_features = None
        last_hidden_state = text_features['last_hidden_state']
        hidden_states = text_features['hidden_states'] if 'hidden_states' in text_features else None
        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":
            last_hidden_state, text_patch_features = last_hidden_state[:, :1], last_hidden_state[:, 1:]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(last_hidden_state.size()).float()
            text_patch_features = last_hidden_state
            last_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, axis=1) / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
            last_hidden_state = last_hidden_state.unsqueeze(1)
        else:
            raise NotImplementedError("Not supported pooling method : %s", self.text_pooling)

        return { 
            'cls_token': last_hidden_state, 
            'patch_tokens': text_patch_features,
            'hidden_states': hidden_states
        }
    
    def _project_and_normalize(self, feats: torch.Tensor, projector: nn.Module, return_raw_and_normed=False):
        """helper function for the forward function of CXRCLIP"""
        if feats == None:
            return None

        assert len(feats.shape) == 3
        feat_embeddings_raw = projector(feats) if projector is not None else feats
        feat_embeddings_normed = feat_embeddings_raw / feat_embeddings_raw.norm(dim=2, keepdim=True)
        
        feat_embeddings_raw, feat_embeddings_normed = feat_embeddings_raw.squeeze(), feat_embeddings_normed.squeeze()
        return (feat_embeddings_raw, feat_embeddings_normed) if return_raw_and_normed else feat_embeddings_normed

    def forward(self, batch, device=None):
        raise NotImplementedError("Subclasses must implement the forward(batch) method.")


class AttentionBasedClip(BaseClip):
    def __init__(self, model_config: Dict, all_loss_config: Dict, tokenizer = None):
        super(AttentionBasedClip, self).__init__(model_config, all_loss_config, tokenizer)


    def flair_attention(self, query_feats, value_feats, value_masks, unit_norm_weighting=False):
        """
        query_feats: [B, 1, D]
        value_feats: [B, P, D]
        value_masks: [B, P, 1]
        unit_norm_weighting: boolean, when the query_feats and value_feats are not normalized.
        return results : [B, 1, D], each D is the weighted summation of the attended features of the value based on a single query features
        """
        results = flair_attention_util(query_feats, value_feats, value_masks, unit_norm_weighting)
        return results

    def equal_weighted_global_local_attention(self):
        """
        attention on the local embeddings and then weighted sum as wl
        then 1/2 cls + 1/2 wl as the final vector.
        """
        return

    def gloria_attention(self, image_feats, text_feats, text_mask):
        """this might need even smaller batch size due to 4d tensor in the similarity matrix"""
        return