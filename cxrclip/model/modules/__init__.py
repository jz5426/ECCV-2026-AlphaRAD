from typing import Dict
import torch
from .image_encoder import XrayDINO, XrayDinov2_224
from .projection import LinearProjectionHead
from .text_encoder import HuggingfaceTextEncoder
from transformers.tokenization_utils import PreTrainedTokenizer

def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "microsoft" and 'xraydino' in config_image_encoder["name"]:
        _image_encoder = XrayDINO(
            ckpt_dir=config_image_encoder['cache_dir'],
            freeze_backbone=config_image_encoder['freeze_backbone'],
            interpolate_pos_encoding=False if config_image_encoder.get('image_size', 518) == 518 else True # default is 518 unless explicitly specificed
        )
    elif config_image_encoder["source"].lower() == "stanford_aiml" and 'xraydinov2_224' in config_image_encoder["name"]:
        _image_encoder = XrayDinov2_224(
            ckpt_dir=config_image_encoder['cache_dir'],
            freeze_backbone=config_image_encoder['freeze_backbone']
        )
    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder

def load_text_encoder(config_text_encoder: Dict, tokenizer: PreTrainedTokenizer):
    if config_text_encoder["source"].lower() == "huggingface":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            tokenizer=tokenizer,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=config_text_encoder["trust_remote_code"],
            dual_cls=config_text_encoder['dual_cls']
        )

    elif config_text_encoder["source"].lower() == "laihaoran":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        revised_dict = None
        if 'carzero_pretrained_dir' in config_text_encoder:
            carzero_best_derived_text_dict = torch.load(config_text_encoder["carzero_pretrained_dir"], map_location="cpu", weights_only=False)
            revised_dict = { key[len('text_encoder.'):] : carzero_best_derived_text_dict[key] for key in carzero_best_derived_text_dict if 'text_encoder.' in key}
            
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            tokenizer=tokenizer,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=config_text_encoder["trust_remote_code"],
            dual_cls=config_text_encoder['dual_cls'],
            custom_pretrain_weights=revised_dict
        )
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    projection_head = LinearProjectionHead(embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"])
    return projection_head