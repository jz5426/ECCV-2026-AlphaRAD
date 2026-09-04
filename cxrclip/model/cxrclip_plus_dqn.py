import logging
from typing import Dict
import torch
from torch import nn
from cxrclip.model.baseclip import AttentionBasedClip
import logging
import numpy as np
import torch.nn.functional as F
from .modules import load_projection_head

log = logging.getLogger(__name__)

class CXRClipPlusWithDQN(AttentionBasedClip):
    """Custom forward function for cxrclip plus with attention AND DQN module. """

    def __init__(self, model_config: Dict, all_loss_config: Dict, config: Dict={}, tokenizer = None):
        super(CXRClipPlusWithDQN, self).__init__(model_config, all_loss_config, tokenizer)
        self.temperature = model_config["temperature"] if "temperature" in model_config else None
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            log.warning("[CXRCLIP] missing temperature scaling factor")

        # for temperory indexing
        self.head_index = None
        if self.head_index is not None and self.head_index >= 0:
            log.warning(f"[CXRClipPlusWithDQN] Make sure in inference mode and this is intentional: head index evaluation {self.head_index}")

        if self.projection:

            # vision
            try:
                if self.vision_dual_cls: # use prompt token classifier if defined
                    self.image_patch_projection = load_projection_head(
                        embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                    ) if model_config['projection_head']['seperate_local_projection_head'] else self.image_cls2_projection
                else: # use text report classifier.
                    self.image_patch_projection = load_projection_head(
                        embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                    ) if model_config['projection_head']['seperate_local_projection_head'] else self.image_cls_projection
            except:
                # backward compatible when the key seperate_local_projection_head does not exists in the ckpt checkpoint.
                self.image_patch_projection = load_projection_head(
                    embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                )

            # text
            try:
                self.word_projection = load_projection_head(
                    embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
                ) if model_config['projection_head']['seperate_local_projection_head'] else self.text_cls_projection
            except:
                self.word_projection = load_projection_head(
                    embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
                )

        # image related tokens for cross attention.
        self.use_all_image_tokens = self.model_config['dqn_image_tokens']['use_all_image_tokens'] if 'dqn_image_tokens' in self.model_config else True

        try:
            self.classification_head = model_config['classification_head']
        except:
            self.classification_head = 'dqn'
        assert self.classification_head in ['dqn', 'radzero', 'both', 'radzero_mh']

        self.enable_radzero_forward_pass, self.enable_dqn_forward_pass, self.enable_multihead_radzero_forward_pass = False, False, False
        if self.classification_head == 'radzero_mh':
            self.enable_multihead_radzero_forward_pass = True
            self.num_radzero_heads = model_config['radzero_head']['number_of_heads']
            self.use_learnable_head_weights = model_config['radzero_head'].get('use_learnable_head_weights', False)
            if self.use_learnable_head_weights:
                # initialized to zeros so softmax starts at uniform (1/H per head)
                self.radzero_head_weights = nn.Parameter(torch.zeros(self.num_radzero_heads))
            log.info(f"[CXRClipPlusWithDQN] use_learnable_head_weights={self.use_learnable_head_weights}, num_radzero_heads={self.num_radzero_heads}")

        self.dqn_fusion_type = model_config['dqn_fusion_type'] if 'dqn_fusion_type' in model_config else None
        self.use_last_n_layer_features = model_config['use_last_n_layer_features'] if 'use_last_n_layer_features' in model_config else [-1]
        self.text_use_last_n_layer_features = list(model_config['text_use_last_n_layer_features']) if 'text_use_last_n_layer_features' in model_config else [-1]
        self.text_cls_type = model_config['cls_type'] if 'cls_type' in model_config else 'cls'
        assert self.text_cls_type in ['patch', 'cls'], 'cls type for text should either be "cls" or "patch".'
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        try:
            self.text_include_cls_token_to_patch_aggregation = model_config['include_cls_token_to_patch_aggregation']
        except:
            self.text_include_cls_token_to_patch_aggregation = False

        try:
            self.radzero_head_temperature = model_config['radzero_head']['temperature']
        except:
            self.radzero_head_temperature = 1
        try:
            self.radzero_head_learnable_temperature = model_config['radzero_head']['learnable_temperature']
        except:
            self.radzero_head_learnable_temperature = False

        # loss temperature for the radzero head
        if self.radzero_head_learnable_temperature:
            self.radzero_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.radzero_head_temperature))
        else:
            self.radzero_logit_scale = torch.tensor(1, dtype=torch.float32)

        try:
            self.radzero_head_enable_bidirectional_cross_attention = model_config['radzero_head']['enable_bidirectional_cross_attention']
        except:
            self.radzero_head_enable_bidirectional_cross_attention = False
        assert self.radzero_head_enable_bidirectional_cross_attention == False, 'Not support this functionality at the moment'

        try:
            self.radzero_head_enable_cross_attend_cls_token = model_config['radzero_head']['enable_cross_attend_cls_token']
        except:
            self.radzero_head_enable_cross_attend_cls_token = False

        # for the loss forwarding
        try:
            self.radzero_head_enable_bce_loss = model_config['radzero_head']['enable_bce_loss']
        except:
            self.radzero_head_enable_bce_loss = False
        try:
            self.radzero_head_enable_softmax_loss = model_config['radzero_head']['enable_softmax_loss']
        except:
            self.radzero_head_enable_softmax_loss = False

    def custom_project_vision_cls_features(self, image_last_cls_features, hidden_states):
        image_cls_embeddings_raw, image_cls_embeddings_for_text = self._project_and_normalize(image_last_cls_features, self.image_cls_projection, return_raw_and_normed=True)
        image_cls_embeddings_raw, image_cls_embeddings_for_text = [image_cls_embeddings_raw], [image_cls_embeddings_for_text]
        return image_cls_embeddings_raw, image_cls_embeddings_for_text

    def custom_project_vision_patch_features(self, image_last_patch_features, hidden_states):
        image_patch_embeddings_raw, image_patch_embeddings_for_text = self._project_and_normalize(image_last_patch_features, self.image_patch_projection, return_raw_and_normed=True)
        image_patch_embeddings_raw, image_patch_embeddings_for_text = [image_patch_embeddings_raw], [image_patch_embeddings_for_text]
        return image_patch_embeddings_raw, image_patch_embeddings_for_text
    
    def custom_project_text_cls_token(self, last_cls_token, hidden_states):
        cls_embeddings_raw, cls_embeddings = self._project_and_normalize(last_cls_token, self.text_cls_projection, return_raw_and_normed=True)
        return cls_embeddings_raw, cls_embeddings

    def custom_project_word_features(self, word_features, hidden_states, attention_mask, ids):
        """
        word_features: of shape [number of query label, number of tokens per query label exclude cls token, feature dimension per token]
        hidden_states: of shape [number of query label, number of tokens per query label + cls token, feature dimension per token]
        attention_mask: of shape [number of query label, number of tokens per query label + cls token]
        """
        word_embeddings_raw, word_embeddings = self._project_and_normalize(word_features, self.word_projection, return_raw_and_normed=True)
        return word_embeddings_raw, word_embeddings

    def multihead_radzero_classification_branch(
        self, 
        image_embeddings, 
        label_embeddings, 
        # label_attention_masks,
        generate_grounding_map=False
    ):
        """
        RadZero-style cosine-similarity cross-attention branch BUT EXPAND TO MULTI-HEAD.
        
        Logic:
        - Head 0: Primary Classification (logits_label_to_image).
        - Heads 1..H: Ensemble Branch (logits_label_to_image_mpnce) & Grounding Map.
        """
        radzero_logit_scale = self.radzero_logit_scale.exp()

        # Standard inputs (Already Projected)
        image_cls = image_embeddings['image_cls_raw_features'] 
        image_patches = image_embeddings['image_patch_raw_features'] 
        assert len(image_cls) == 1 and len(image_patches) == 1, 'currently only support single layer'
        image_cls, image_patches = image_cls[0], image_patches[0]
        label_cls = label_embeddings['text_cls_raw_features'] 

        # Build image keys/values
        if self.radzero_head_enable_cross_attend_cls_token:
            image_kv = torch.cat([image_cls.unsqueeze(1), image_patches], dim=1)  # [B, 1+P, D]
        else:
            image_kv = image_patches  # [B, P, D]

        B, K, Total_D = image_kv.shape
        Q, _ = label_cls.shape
        H = self.num_radzero_heads
        
        # Safety Check
        assert Total_D % H == 0, f"Feature dim {Total_D} is not divisible by num_heads {H}"
        D_head = Total_D // H  
        
        # 1. Split into Heads for both image and text (Reshape)
        # image_kv: [B, K, Total_D] -> [B, K, H, D_head]
        image_kv_h = image_kv.view(B, K, H, D_head)
        
        # label_cls: [Q, Total_D] -> [Q, H, D_head]
        label_cls_h = label_cls.view(Q, H, D_head)

        # 2. Normalize Per Head (Subspace L2 Normalization)
        image_kv_h = F.normalize(image_kv_h, p=2, dim=-1)
        label_cls_h = F.normalize(label_cls_h, p=2, dim=-1)

        # 3. Calculate Attention Logits (Similarity) Per Head
        # [B, K, H, D_head] vs [Q, H, D_head] -> [B, Q, K, H]
        attention_logits_h = torch.einsum("bkhd,qhd->bqkh", image_kv_h, label_cls_h)
        
        # Scale by temperature before softmax
        attention_logits_h_scaled = attention_logits_h * radzero_logit_scale

        # 4. Softmax Per Head
        # attn_weights_h: [B, Q, K, H]
        attn_weights_h = F.softmax(attention_logits_h_scaled, dim=2)

        # 5. Weighted Sum (Context) Per Head
        # [B, Q, K, H] * [B, K, H, D_head] -> [B, Q, H, D_head]
        image_context_h = torch.einsum("bqkh,bkhd->bqhd", attn_weights_h, image_kv_h)

        # 6. Normalize Context Per Head 
        image_context_h = F.normalize(image_context_h, p=2, dim=-1)
        # assert image_context_h.shape[-1] == 768, 'This is not expected, all head should have dimension 768'

        # 7. Final Logits Per Head
        # [B, Q, H, D_head] vs [Q, H, D_head] -> [B, Q, H]
        logits_per_head = torch.einsum("bqhd,qhd->bqh", image_context_h, label_cls_h)
        
        # Scale final logits for the Loss function
        logits_per_head = logits_per_head * radzero_logit_scale

        # 8.ENSEMBLING THE HEADS FOR BCE AND DEEP SUPERVISION FOR EACH HEAD
        # pre-compute once; shared by both classification and grounding aggregation below
        head_weights = torch.softmax(self.radzero_head_weights, dim=0) if self.use_learnable_head_weights else None  # [H] or None

        # always computed as a sanity reference regardless of aggregation mode
        mean_logits = logits_per_head.mean(-1)  # [B, Q]

        if self.head_index is not None and self.head_index >= 0:
            logits_label_to_image = logits_per_head[..., self.head_index]  # [B, Q]
        elif head_weights is not None:
            logits_label_to_image = (logits_per_head * head_weights).sum(-1)  # [B, Q]
            assert logits_label_to_image.shape == mean_logits.shape, \
                f"shape mismatch: weighted {logits_label_to_image.shape} vs mean {mean_logits.shape}"
        else:
            logits_label_to_image = mean_logits

        logits_label_to_image_mpnce = logits_per_head

        grounding_stuffs = {}
        if generate_grounding_map:
            avg_attention_logits = attention_logits_h_scaled.mean(dim=-1)  # [B, Q, K]
            avg_attn_weights = attn_weights_h.mean(dim=-1)                 # [B, Q, K]
            grounding_stuffs = {
                "grounding_map": avg_attention_logits, 
                "grounding_logits": avg_attn_weights,
                'grounding_per_head': attention_logits_h_scaled # used for check what the subspace learns.
            }

        results = {
            # Primary Head (Head 0) -> for alphaBCE
            'logits_label_to_image': logits_label_to_image.unsqueeze(-1),
            # Ensemble Head (Heads 1..N) - for MPNCE 
            'logits_label_to_image_mpnce': logits_label_to_image_mpnce,
            'logits_image_to_label': None,
            'radzero_head_enable_bce_loss': self.radzero_head_enable_bce_loss,
            'radzero_head_enable_softmax_loss': self.radzero_head_enable_softmax_loss,
            'radzero_logit_scale': radzero_logit_scale,
            'radzero_enriched_image_context': image_context_h
        }
        return results | grounding_stuffs

    def get_grounding_maps(self, image_patch_normed, label_cls_normed):
        """
        image_patch_normed: normalized image patch features (without cls)
        label_cls_normed: normalized label cls feature
        """
        # cosine similarity between each label CLS and each patch token
        # [B, Q, P] = [B, P, D] @ [Q, D]^T
        grounding_logits = torch.einsum("bpd,qd->bqp", image_patch_normed, label_cls_normed)
        # softmax across patches -> attention distribution over spatial tokens
        grounding_map = F.softmax(grounding_logits, dim=-1)  # [B, Q, P]
        return grounding_map, grounding_logits
    
