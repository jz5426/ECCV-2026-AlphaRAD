from torch.nn import functional as F
import torch
import logging
import random
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    log.info("Global seed set to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def map_column(tag):
    if 'AP' in tag:
        return 'AP'
    if 'PA' in tag:
        return 'PA'
    if 'image' in tag:
        return tag
    return 'Lateral'

def curate_dqn_input_labels(
    label: str, 
    prompt_template: str,
    observation_explanations: dict = {}, 
    anonymize_observation_explanation: bool = False,
    random_description_selection: bool = False,
    probability_for_selecting_description=0
):  
    label = label.lower()
    descriptions = observation_explanations.get(label, [])

    # single_description = label
    if len(observation_explanations) > 0:
        if label == 'no findings':
            single_description = label #+ '.'
        else:
            assert len(descriptions) > 0, f'Each disease should has a description but {label} does not have.'
            single_description = descriptions[random.randint(0, len(descriptions)-1)] if random_description_selection else descriptions[0]
            # remove the 
            if single_description[-1] == '.':
                single_description = single_description[:-1]
    else:
        probability_for_selecting_description = 0
    
    # randomly pick description or label base on probability
    x = np.random.choice([1, 2], p=[probability_for_selecting_description, 1-probability_for_selecting_description])
    if x == 1: # keep only the description of the entity
        final_label = prompt_template.format(single_description.lower()) # newly modified.
    elif x == 2: # keep only the entity names (most frequently use)
        final_label = prompt_template.format(label) # if not label.lower().startswith('there is') else label # NOTE: this case mainly handles ms-cxr

    return final_label.strip()

def anonymize_qwen30b_204disease_descriptions(observation_explanations):
    for key in observation_explanations:
        desp = observation_explanations[key]
        # anaoymize the disease name and only keep the disease explanation
        marker_idx = desp.lower().find('appears')
        assert marker_idx >= 0, 'all sentences should have the word "appears"'
        desp = desp[marker_idx:].lower().replace(key.lower(), '__') # last resort
        observation_explanations[key] = desp
    return observation_explanations


def flair_attention_util(query_feats, value_feats, value_masks, unit_norm_weighting=False):
    if value_masks is not None:
        assert value_feats.shape[1] == value_masks.shape[1], "the number of tokens should be the same between the masks and the number of tokens"

    # --- Normalize for cosine similarity attention ---
    q_norm, v_norm = query_feats, value_feats
    if unit_norm_weighting:
        q_norm = F.normalize(query_feats, dim=-1)
        v_norm = F.normalize(value_feats, dim=-1)

    # Compute cosine similarity attention scores: [B, P], attention score for each value tokens, for each query
    attn_scores = torch.sum(q_norm * v_norm, dim=-1)  # [B, P]

    if value_masks is not None:
        # Apply mask (0 for invalid, 1 for valid positions)
        attn_scores = attn_scores.masked_fill(value_masks.squeeze(-1) == 0, float('-inf'))

    # Softmax over valid positions
    attn_weights = F.softmax(attn_scores, dim=-1)  # [B, P]

    if value_masks is not None:
        # Zero out attention weights where masked
        attn_weights = attn_weights * value_masks.squeeze(-1)  # [B, P]

    # Weighted sum: [B, P, 1]X[B,P, D]=[B, 1, D]
    results = torch.sum(attn_weights.unsqueeze(-1) * value_feats, dim=1, keepdim=True)

    return results.squeeze()