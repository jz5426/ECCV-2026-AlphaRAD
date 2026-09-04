from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer
from .cxrclip_plus_dqn import CXRClipPlusWithDQN

def build_model(model_config: Dict, loss_config: Dict, config: Dict={}, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    model = CXRClipPlusWithDQN(model_config, loss_config, config, tokenizer)
    return model
