from typing import Callable
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from transformers import AutoImageProcessor
from transformers import AutoModel, AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from cxrclip.model.modules.dinov2_utils import (
    LayerScale
)
import logging
log = logging.getLogger(__name__)

TypeClsToken = Float[Tensor, "batch_size embed_dim"]
TypePatchTokensFlat = Float[Tensor, "batch_size (height width) embed_dim"]
TypePatchTokens = Float[Tensor, "batch_size embed_dim height width"]
TypeInputImages = Tensor

class XrayDINOLocal(nn.Module):

    def __init__(self, ckpt_path, freeze_backbone=False, interpolate_pos_encoding=False):
        super().__init__()
        config = AutoConfig.from_pretrained(ckpt_path, local_files_only=True)
        config.output_hidden_states = True  # need this to access the intermediate layer outputs

        self.model = AutoModel.from_pretrained(ckpt_path, config=config, local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(ckpt_path, use_fast=False, local_files_only=True)
        self.interpolate_pos_encoding = interpolate_pos_encoding
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode(self, inputs: BatchFeature) -> tuple[TypeClsToken, TypePatchTokensFlat]:
        if self.interpolate_pos_encoding:
            # for 224 resolution or 256 resolution (non-518)
            outputs = self.model(inputs, output_hidden_states=True, interpolate_pos_encoding=True)
        else:
            # for 518 resolution
            outputs = self.model(inputs, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 1:]

        return cls_token, patch_tokens, outputs.hidden_states

    def reshape_patch_tokens(
        self,
        patch_tokens_flat: TypePatchTokensFlat,
    ) -> TypePatchTokens:
        input_size = self.processor.crop_size["height"]
        patch_size = self.model.config.patch_size
        embeddings_size = input_size // patch_size
        patches_grid = rearrange(
            patch_tokens_flat,
            "batch (height width) embed_dim -> batch embed_dim height width",
            height=embeddings_size,
        )
        return patches_grid

    def extract_features(
        self,
        inputs: TypeInputImages,
    ) -> tuple[TypeClsToken, TypePatchTokens]:
        cls_token, patch_tokens, hidden_states = self.encode(inputs)
        return cls_token, patch_tokens, hidden_states

    def extract_cls_token(self, image_or_images: TypeInputImages) -> TypeClsToken:
        cls_token, _ = self.extract_features(image_or_images)
        return cls_token

    def extract_patch_tokens(self, image_or_images: TypeInputImages) -> TypePatchTokens:
        _, patch_tokens = self.extract_features(image_or_images)
        return patch_tokens

    def forward(self, *args) -> tuple[TypeClsToken, TypePatchTokens]:
        return self.extract_features(*args)

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, nn.Conv2d):
        module.reset_parameters()

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module