import logging
from typing import Dict, Union, List
import numpy as np
import torch
from scipy.special import softmax
from sklearn import metrics
from tqdm import tqdm
from cxrclip.evaluator_utils import *
from cxrclip.util.utils import anonymize_qwen30b_204disease_descriptions, curate_dqn_input_labels

log = logging.getLogger(__name__)

class BaseEvaluator:
    """
    for evaluation that are not reloading the weights, during training.
    """
    def __init__(self, config: Dict, rank=None):
        self.config = config
        self.rank = rank

        # Determine device
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # DDP mode: use provided rank (e.g., local_rank)
            assert rank is not None, "In DDP mode, 'rank' must be provided."
            self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        else:
            # Non-DDP: use default CUDA if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Device: {self.device}')

        if 'eval_method' in self.config['designs']:
            self.similarity_type = self.config['designs']['eval_method']
        else:
            try:
                self.similarity_type = self.ckpt_config['designs']['loss']['filip']['variant']
            except:
                self.similarity_type = 'cosine' # cosine

        # set by subclasses from the eval-time config; None means don't save
        self.seg_save_dir = None

        self.keep_prompt_queried_image_embeddings = False
        if 'loss' in self.config['designs'] and 'keep_prompt_queried_image_embeddings' in self.config['designs']['loss']:
            self.keep_prompt_queried_image_embeddings = self.config['designs']['loss']['keep_prompt_queried_image_embeddings']

        try:
            self.use_all_image_tokens = self.config['designs']['model']['dqn_image_tokens']['use_all_image_tokens']
        except:
            self.use_all_image_tokens = True # default

        try:
            self.anonymize_observation_explanation = False
            self.observation_explanation = dict({})
            if 'anonymize_observation_explanation' in self.config['designs']['data'] and self.config['designs']['data']['anonymize_observation_explanation'] is not None:
                self.anonymize_observation_explanation = True
                self.observation_explanation = anonymize_qwen30b_204disease_descriptions(self.observation_explanation)

            self.prompt_template = "There is {}." if 'add_alignment_prompt_prefix' in self.config["designs"]["data"] and self.config["designs"]["data"]["add_alignment_prompt_prefix"] else "{}."

            # determine what zero-shot inference method to use.
            self.zero_shot_method = 'zeroshot_binary'
            if 'zero_shot_method' in self.config['designs']:
                self.zero_shot_method = self.config['designs']['zero_shot_method']
            assert self.zero_shot_method in ['zeroshot_binary', 'zeroshot_binary_dqn']

            self.enable_enriched_embeddings_as_input_to_dqn = (
                self.config['model']['enable_enriched_embeddings_as_input_to_dqn'] if 'enable_enriched_embeddings_as_input_to_dqn' in self.config['model'] 
                else False
            )
            assert self.enable_enriched_embeddings_as_input_to_dqn == False, 'This should be false to match the inference distribution.'
            
            # evaluation flags for dqn based
            if 'dqn' in self.zero_shot_method:
                try:
                    self.dqn_similarity = self.config["designs"]['loss']['cxrclip_plus_dqn']['loss_type']
                except:
                    self.dqn_similarity = 'BCEWithLogitsLoss'
                
                if self.dqn_similarity == 'CrossEntropyLoss':
                    try:
                        self.dqn_is_bidirectional = self.config["designs"]['loss']['cxrclip_plus_dqn']['include_bidirectional_modality2label_mp_infoNCE']
                    except:
                        self.dqn_is_bidirectional = False
                else:
                    try:
                        self.dqn_is_bidirectional = self.config['model']['enable_dqn_prediction_by_image_query']
                    except:
                        self.dqn_is_bidirectional = False
        except:
            log.warning('[BaseEvaluator] make sure this is currently in fewshot finetuning mode')

    def on_the_fly_evaluation_one_data(self, dataset_name, similarity_type):
        dataloader = self.data_loader_dict[dataset_name]
        class_list = get_class_list(dataset_name)
        predictions, attention_maps, attention_maps_perhead, enriched_img_embeddings = None, None, None, None
        label_names, gt_box_labels, image_paths, masks = [], [], [], []
        for idx, batch in tqdm(enumerate(dataloader), desc=f"[{dataset_name}]", total=len(dataloader)):

            image_results = self.encode_image(batch["images"])
            
            image_cls_embeddings, image_cls2_embeddings, image_patch_embeddings = image_results['cls_token'], image_results['cls2_token'], image_results['patch_tokens']
            image_cls_embeddings_raw, image_cls2_embeddings_raw, image_patch_embeddings_raw = image_results['cls_token_raw'], image_results['cls2_token_raw'], image_results['patch_tokens_raw']

            batch_label_names = batch["label_names"] if "label_names" in batch and batch['label_names'] is not None else np.array([])
            label_names.extend(batch_label_names)
            if 'image_paths' in batch:
                image_paths.extend(batch["image_paths"]) # for pointing game task

            if 'masks' in batch:
                masks.extend(batch["masks"]) # for pointing game task
            if "boxes" in batch:
                gt_box_labels.extend(batch['boxes'])

            # per batch evaluation.
            if dataset_name in {
                'chestxdet10_gt', 
                'vindr_cxr_gr', 
                'node21',
                'chexlocalize',
                'ms_cxr_test', 
                'ms_cxr_test_v2', 
                'padchest_gr',
                'siim_pneumothorax_seg', 
                'chexlocalize_seg',
                'qatacovid_seg',
            }:
                batch_attention_maps, batch_attention_maps_perhead = self._zeroshot_pointinggame_dqn_helper(
                    image_cls2_embeddings_raw if image_cls2_embeddings_raw is not None else image_cls_embeddings_raw, 
                    image_cls2_embeddings if image_cls2_embeddings is not None else image_cls_embeddings,
                    image_patch_embeddings_raw,
                    image_patch_embeddings,
                    batch_label_names, 
                    class_list
                )
                attention_maps = np.concatenate([attention_maps, batch_attention_maps], axis=0) if attention_maps is not None else batch_attention_maps
                attention_maps_perhead = np.concatenate([attention_maps_perhead, batch_attention_maps_perhead], axis=0) if attention_maps_perhead is not None else batch_attention_maps_perhead

            if dataset_name in {
                "siim_pneumothorax", 
                'rsna_pneumonia', 
                "vindr_cxr", 
                "shenzhenxray",
                "cxrlt_task3",
                "cxrlt_task2",
                "montgomery",
                "chest14", 
                "chexpert", 
                "chexpert5x200",
                "physician_padchest5",
                "physician_padchest207",
                "chexchonet_composite_slvh_dlv",
                "chexchonet_slvh",
                "chexchonet_dlv",
                "chestxdet10",
                "covidkaggle"
            }:
                if self.zero_shot_method == 'zeroshot_binary_dqn':
                    batch_predictions, batch_normed_enriched_img_embeddings = self._zeroshot_binary_dqn_helper(
                        image_cls2_embeddings_raw if image_cls2_embeddings_raw is not None else image_cls_embeddings_raw, 
                        image_cls2_embeddings if image_cls2_embeddings is not None else image_cls_embeddings,
                        image_patch_embeddings_raw,
                        image_patch_embeddings,
                        batch_label_names, 
                        class_list
                    )
                    predictions = np.concatenate([predictions, batch_predictions], axis=0) if predictions is not None else batch_predictions
                    enriched_img_embeddings = np.concatenate([enriched_img_embeddings, batch_normed_enriched_img_embeddings], axis=0) if enriched_img_embeddings is not None else batch_normed_enriched_img_embeddings

        results = {}
        if len(label_names) > 0 and type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]

        # disease label grounding game results
        if dataset_name in {
            'chestxdet10_gt', 
            'vindr_cxr_gr', 
            'chexlocalize', 
            'node21',
        }:
            pg_results = gather_pointinggame_statistics(
                class_list, 
                attention_maps, 
                gt_box_labels, 
                label_names,
                image_paths,
                grounding_type=get_grounding_type(dataset_name),
                visual_save_dir=None,
                draw_box_on_overlay=True
            )
            results['zeroshot_pointing_game'] = pointing_game_score(pg_results)

        # phrase grounding game results
        if dataset_name in {'ms_cxr_test', 'ms_cxr_test_v2', 'padchest_gr'}:
            pg_results = gather_pointinggame_statistics(
                class_list, 
                attention_maps, 
                gt_box_labels, 
                label_names,
                image_paths,
                grounding_type=get_grounding_type(dataset_name),
                visual_save_dir=None,
                draw_box_on_overlay=True
            )
            results['zeroshot_refer_grounding'] = refer_grounding_score(pg_results)

        # zeroshot segmentation results
        if dataset_name in {'siim_pneumothorax_seg', 'rsna_pneumonia_seg', 'qatacovid_seg', 'chexlocalize_seg'}:
            seg_save_dir = (f'{self.seg_save_dir}/{dataset_name}'
                            if self.seg_save_dir is not None else None)
            seg_results = gather_seg_statistics(
                attention_maps,
                image_paths,
                masks,
                label_names,
                class_list,
                save_dir=seg_save_dir
            )
            results['zeroshot_seg'] = seg_score(seg_results)

        if dataset_name in {
            "siim_pneumothorax", 
            'rsna_pneumonia', 
            "vindr_cxr", 
            "chest14", 
            "chexpert", 
            "shenzhenxray",
            "cxrlt_task3",
            "cxrlt_task2",
            "montgomery",
            "chexchonet_composite_slvh_dlv",
            "chexchonet_slvh",
            "chexchonet_dlv",
            "chexpert5x200",
            "physician_padchest207",
            "chestxdet10",
            "covidkaggle"
        }:
            if self.zero_shot_method == 'zeroshot_binary_dqn':
                zs_results, class_counts = self.gather_zeroshot_dqn_classification_statistics(class_list, predictions, label_names)
                results['zeroshot_binary'] = classification_score(zs_results, class_counts)

        return results

    def evaluate_clip_online(self):

        eval_results = {
            'zeroshot': [], # contains dictionary results for stats
            'retrieval': []
        }
        self.model.eval()
        with torch.no_grad():
            for dataset_name in self.data_loader_dict:
                results = self.on_the_fly_evaluation_one_data(dataset_name, self.similarity_type)
                if dataset_name in {
                    "siim_pneumothorax", 
                    "rsna_pneumonia", 
                    "vindr_cxr", 
                    "shenzhenxray",
                    "cxrlt_task3",
                    "cxrlt_task2",
                    "montgomery",
                    "chexchonet_composite_slvh_dlv",
                    "chexchonet_slvh",
                    "chexchonet_dlv",
                    "chest14", 
                    "chexpert", 
                    "chexpert5x200", 
                    "physician_padchest207",
                    "chestxdet10",
                    "covidkaggle",
                }:
                    eval_results['zeroshot'].append((dataset_name, results["zeroshot_binary"]))
                if dataset_name in {"mimic_cxr", "openi"}:
                    eval_results['retrieval'].append((dataset_name, results["retrieval_i2t"]))

        return eval_results

    def zeroshot_binary(
        self, 
        image_embeddings: np.ndarray, 
        image_patch_embeddings: np.ndarray,
        label_names: list, 
        class_list: list
        ):
        log.info("evaluate zero-shot binary classification via zeroshot_binary method")
        if type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]
        predictions = self._zeroshot_binary_helper(
            image_embeddings, 
            image_patch_embeddings,
            label_names, 
            class_list
        )
        results = self.gather_zeroshot_binary_classification_statistics(class_list, predictions, label_names)
        return classification_score(results)

    def zeroshot_binary_dqn(
        self, 
        image_embeddings_raw: np.ndarray, 
        image_patch_embeddings_raw: np.ndarray,
        label_names: list, 
        class_list: list,
        batch_size = 32
        ):
        log.info("evaluate zero-shot binary classification via zeroshot_binary DQN")
        if type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]

        predictions, _ = self._zeroshot_binary_dqn_helper(
            image_embeddings_raw,
            image_patch_embeddings_raw,
            label_names,
            class_list,
            batch_size
        )

        results, class_counts = self.gather_zeroshot_dqn_classification_statistics(class_list, predictions, label_names)
        return classification_score(results)
    
    def encode_image(self, image: torch.Tensor):
        # unwrap safely if using DDP
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        with torch.no_grad():
            results = model.encode_image(image.to(self.device), model.use_last_n_layer_features) # check baseclip
            cls_tokens, patch_emb = results['cls_token'], results['patch_tokens'], 
            img_cls_emb, img_cls2_emb = cls_tokens[:, :1, :], cls_tokens[:, 1:2, :] if cls_tokens.shape[1] == 2 else None, 

            # make projection if applicable
            img_cls2_emb_raw = None
            if model.projection and len(model.use_last_n_layer_features) == 1:
                img_cls_emb_raw, img_cls_emb = model._project_and_normalize(img_cls_emb, model.image_cls_projection, return_raw_and_normed=True)
                if img_cls2_emb is not None:
                    img_cls2_emb_raw, img_cls2_emb = model._project_and_normalize(img_cls2_emb, model.image_cls2_projection, return_raw_and_normed=True)
                patch_emb_raw, patch_emb = model._project_and_normalize(patch_emb, model.image_patch_projection, return_raw_and_normed=True)

                img_cls_emb_raw, img_cls_emb = [img_cls_emb_raw], [img_cls_emb]
                patch_emb_raw, patch_emb = [patch_emb_raw], [patch_emb]
            elif model.projection:
                # take into account the deep dqn setting
                img_cls_emb_raw, img_cls_emb = model.custom_project_vision_cls_features(img_cls_emb, results['hidden_states'])
                patch_emb_raw, patch_emb = model.custom_project_vision_patch_features(patch_emb, results['hidden_states'])
                if img_cls2_emb is not None:
                    assert False, "This functionality is not supported at the moment."

        cls_token = [t.detach().cpu().numpy() for t in img_cls_emb]
        cls2_token = [t.detach().cpu().numpy() for t in img_cls2_emb] if img_cls2_emb is not None else None
        patch_tokens = [t.detach().cpu().numpy() for t in patch_emb]

        cls_token_raw = [t.detach().cpu().numpy() for t in img_cls_emb_raw]
        cls2_token_raw = [t.detach().cpu().numpy() for t in img_cls2_emb_raw] if img_cls2_emb_raw is not None else None
        patch_tokens_raw = [t.detach().cpu().numpy() for t in patch_emb_raw]
        
        return {
            'cls_token': cls_token,
            'cls2_token': cls2_token,
            'patch_tokens': patch_tokens,
            'cls_token_raw': cls_token_raw,
            'cls2_token_raw': cls2_token_raw,
            'patch_tokens_raw': patch_tokens_raw
        }

    def encode_text(self, text_token: Union[str, List[str], Dict, torch.Tensor], text_max_length):

        # if it is not encoded, encode it now.
        if isinstance(text_token, str) or isinstance(text_token, list):
            text_token = self.datamodule.tokenizer(
                text_token, padding="max_length", truncation=True, 
                return_tensors="pt", max_length=text_max_length + 1 if self.datamodule.tokenizer.dual_cls else text_max_length
            )
        # unwrap safely if using DDP
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        attention_masks = text_token['attention_mask'].to(self.device)

        with torch.no_grad():
            text_results = model.encode_text(text_token.to(self.device), model.text_use_last_n_layer_features) # check baseclip
            cls_emb, word_emb, hidden_states = text_results['cls_token'], text_results['patch_tokens'], text_results['hidden_states']

            # depends on the config, toggle the cls embedding (patch based or cls based or layer-wise aggregate and etc)
            if model.text_cls_type == 'cls':
                cls_emb_raw, cls_emb = model.custom_project_text_cls_token(cls_emb, hidden_states)
            else:
                cls_emb_raw, cls_emb = model.custom_project_word_features(word_emb, hidden_states, attention_masks, text_token['input_ids'])

            word_emb_raw, word_emb = model._project_and_normalize(word_emb, model.word_projection, return_raw_and_normed=True)

        return {
            'cls_emb': cls_emb.detach().cpu().numpy(),
            'word_emb': word_emb.detach().cpu().numpy(),
            'cls_emb_raw': cls_emb_raw.detach().cpu().numpy(),
            'word_emb_raw': word_emb_raw.detach().cpu().numpy(),
            'attention_mask': text_token['attention_mask'].detach().cpu().numpy(),
        }

    def _zeroshot_binary_helper(
        self, 
        image_embeddings: np.ndarray, 
        image_patch_embeddings: np.ndarray,
        label_names: list, 
        class_list: list
        ):
        result = {}
        predictions = []
        for class_name in class_list:
            l = disease_name_mapping(class_name)
            des = curate_dqn_input_labels(l, self.prompt_template, self.observation_explanation, self.anonymize_observation_explanation)
            prompts = [
                f"There is no {disease_name_mapping(class_name)}." if len(des) == 0 else f"There is not {disease_name_mapping(class_name)}, which {des}",
                f"There is {disease_name_mapping(class_name)}." if len(des) == 0 else f"There is {disease_name_mapping(class_name)}, which {des}"
            ]

            results = self.encode_text(
                prompts,
                self.config["base"]["prompt_max_length"] if 'prompt_max_length' in self.config["base"] else self.config["base"]["text_max_length"]
            )
            text_embeddings = results['cls_emb']

            if 'attention' in self.similarity_type and self.keep_prompt_queried_image_embeddings:
                similarities = text_guided_similarities(image_embeddings, image_patch_embeddings, text_embeddings)
            else:
                similarities = image_embeddings @ text_embeddings.T

            similarities = softmax(similarities, axis=1)

            predictions.append(similarities)
        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def aggregate_attention_maps(self, layer_attention_maps, method='entropy'):
        """
        layer_attention_maps: list of tensor of shape [batch size, heads (optional), query tokens, patch tokens]
        """
        assert method in ['mean', 'entropy', 'top-k']
        aggregated_attention_maps = []
        for weighted_attention in layer_attention_maps:
            aggregated_attention_maps.append(weighted_attention.mean(dim=1))

        return aggregated_attention_maps

    def _zeroshot_pointinggame_dqn_helper(
        self, 
        image_cls_embeddings_raw: List[np.ndarray], 
        image_cls_embeddings_normed: List[np.ndarray], 
        image_patch_embeddings_raw: List[np.ndarray],
        image_patch_embeddings_normed: List[np.ndarray],
        label_names: list, 
        class_list: list,
        batch_size = 32
    ):
        # unwrap safely if using DDP
        model = self.model.module if hasattr(self.model, "module") else self.model
        device = 'cuda' #next(model.dqn.parameters()).device
        class_list = [disease_name_mapping(l) for l in class_list]
        model.eval()
        with torch.no_grad():
            # format label with description
            final_labels = []
            for l in class_list:
                # l = disease_name_mapping(l) # already di in line 597
                label = curate_dqn_input_labels(l, self.prompt_template, self.observation_explanation, self.anonymize_observation_explanation)
                final_labels.append(label)
            text_results = self.encode_text(final_labels, text_max_length=self.config["base"]["dqn_label_max_length"])
            class_label_cls_features_raw, class_label_cls_features_normed = text_results['cls_emb_raw'], text_results['cls_emb']
            class_label_cls_features_raw = class_label_cls_features_raw[None, :] if len(class_label_cls_features_raw.shape) == 1 else class_label_cls_features_raw
            class_label_patch_features_raw, class_label_patch_features_normed = text_results['word_emb_raw'], text_results['word_emb']
            class_label_patch_features_raw = class_label_patch_features_raw[None, :, :] if len(class_label_patch_features_raw.shape) == 2 else class_label_patch_features_raw
            class_label_attention_mask = text_results['attention_mask']

            # image_features_raw = np.concatenate((image_cls_embeddings_raw[:, None, :], image_patch_embeddings_raw), axis=1)
            class_label_cls_features_raw = torch.tensor(class_label_cls_features_raw, device=device)
            attention_maps, attention_maps_perhead = [], []
            image_features_raw = [
                torch.tensor(np.concatenate([t1[:, None, :] if len(t1.shape) == 2 else t1[None, None, :], t2 if len(t2.shape) == 3 else t2[None, :, :]], axis=1), device=device) 
                for t1, t2 in zip(image_cls_embeddings_raw, image_patch_embeddings_raw)
            ]
            image_cls_embeddings_normed = [torch.tensor(layer_feat, device=device) for layer_feat in image_cls_embeddings_normed]
            image_patch_embeddings_normed = [torch.tensor(layer_feat, device=device) for layer_feat in image_patch_embeddings_normed]
            
            class_label_cls_features_normed = torch.tensor(class_label_cls_features_normed, device=device)
            class_label_patch_features_raw = torch.tensor(class_label_patch_features_raw, device=device)
            class_label_patch_features_normed = torch.tensor(class_label_patch_features_normed, device=device)
            class_label_attention_mask = torch.tensor(class_label_attention_mask, device=device)

            for i in range(0, image_features_raw[0].shape[0], batch_size):

                batch_image_features_raw = [layer_feat[i:i + batch_size, :, :] for layer_feat in image_features_raw]

                if not self.use_all_image_tokens:
                    batch_image_features_raw = [layer_feat[:, 1:, ] for layer_feat in batch_image_features_raw]

                batch_attention_maps, batch_attention_maps_perhead = None, None
                if model.enable_multihead_radzero_forward_pass:
                    results = model.multihead_radzero_classification_branch(
                        image_embeddings={
                            'image_cls_raw_features': [image_feat[:, 0, :] for image_feat in batch_image_features_raw],
                            'image_patch_raw_features': [image_feat[:, 1:, :] for image_feat in batch_image_features_raw]
                        },
                        label_embeddings={
                            'text_cls_raw_features': class_label_cls_features_raw,
                        },
                        generate_grounding_map=True
                    )
                    batch_attention_maps = results['grounding_map'] 
                    batch_attention_maps_perhead = results['grounding_per_head']
                    batch_attention_maps = [batch_attention_maps.unsqueeze(1)]

                # original vanilla average solution.
                aggregated_attention_maps = self.aggregate_attention_maps(batch_attention_maps, method='mean')
                attention_maps.append(torch.stack(aggregated_attention_maps, dim=0).mean(dim=0).cpu().numpy())

                if batch_attention_maps_perhead is not None:
                    attention_maps_perhead.append(batch_attention_maps_perhead.detach().cpu().numpy())
            attention_maps = np.concatenate(attention_maps, axis=0)
            attention_maps_perhead = np.concatenate(attention_maps_perhead, axis=0) if len(attention_maps_perhead) > 0 else None

        return attention_maps, attention_maps_perhead
    
    def _zeroshot_binary_dqn_helper(
            self, 
            image_cls_embeddings_raw: List[np.ndarray], 
            image_cls_embeddings_normed: List[np.ndarray],
            image_patch_embeddings_raw: List[np.ndarray],
            image_patch_embeddings_normed: List[np.ndarray],
            label_names: list, 
            class_list: list,
            batch_size = 32
        ):

        # unwrap safely if using DDP
        model = self.model.module if hasattr(self.model, "module") else self.model
        device = 'cuda' #next(model.dqn.parameters()).device
        class_list = [disease_name_mapping(l) for l in class_list]
        model.eval()
        with torch.no_grad():
            # format label with description
            final_labels = []
            for l in class_list:
                l = disease_name_mapping(l)
                label = curate_dqn_input_labels(l, self.prompt_template, self.observation_explanation, self.anonymize_observation_explanation)
                final_labels.append(label)
            text_results = self.encode_text(final_labels, text_max_length=self.config["base"]["dqn_label_max_length"])
            class_label_cls_features_raw, class_label_cls_features_normed = text_results['cls_emb_raw'], text_results['cls_emb']
            class_label_cls_features_raw = class_label_cls_features_raw[None, :] if len(class_label_cls_features_raw.shape) == 1 else class_label_cls_features_raw
            class_label_cls_features_normed = class_label_cls_features_normed[None, :] if len(class_label_cls_features_normed.shape) == 1 else class_label_cls_features_normed
            
            class_label_patch_features_raw, class_label_patch_features_normed = text_results['word_emb_raw'], text_results['word_emb']
            class_label_patch_features_raw = class_label_patch_features_raw[None, :, :] if len(class_label_patch_features_raw.shape) == 2 else class_label_patch_features_raw
            class_label_patch_features_normed = class_label_patch_features_normed[None, :, :] if len(class_label_patch_features_normed.shape) == 2 else class_label_patch_features_normed
            class_label_attention_mask = text_results['attention_mask']

            image_features_raw = [np.concatenate([t1[:, None, :], t2], axis=1) for t1, t2 in zip(image_cls_embeddings_raw, image_patch_embeddings_raw)]

            class_cls_features_raw = torch.tensor(class_label_cls_features_raw, device=device)
            class_label_patch_features_normed = torch.tensor(class_label_patch_features_normed, device=device)
            class_label_attention_mask = torch.tensor(class_label_attention_mask, device=device)
            predictions = []

            # feedforward the fusion module for prediction.
            image_features_raw = [torch.tensor(layer_feat, device=device) for layer_feat in image_features_raw]
            image_cls_embeddings_normed = [torch.tensor(layer_feat, device=device) for layer_feat in image_cls_embeddings_normed]
            image_patch_embeddings_normed = [torch.tensor(layer_feat, device=device) for layer_feat in image_patch_embeddings_normed]
            class_label_patch_features_raw = [torch.tensor(layer_feat, device=device) for layer_feat in class_label_patch_features_raw]

            image_embeddings = []
            for i in range(0, image_features_raw[0].shape[0], batch_size):

                batch_image_features_raw = [layer_feat[i:i + batch_size, :, :] for layer_feat in image_features_raw]
                if not self.use_all_image_tokens:
                    batch_image_features_raw = [layer_feat[:, 1:, ] for layer_feat in batch_image_features_raw]

                t2i_preds, i2t_preds = 0, 0
                enriched_image_embeddings = None
                if model.enable_radzero_forward_pass:
                    results = model.radzero_classification_branch(
                        image_embeddings={
                            "image_cls_raw_features": [image_feat[:, 0, :] for image_feat in batch_image_features_raw],
                            "image_patch_raw_features": [image_feat[:, 1:, :] for image_feat in batch_image_features_raw]
                        },
                        label_embeddings={
                            "text_cls_raw_features": class_cls_features_raw,
                        },
                    )
                    t2i_preds = results['logits_label_to_image']
                    enriched_image_embeddings = results['radzero_enriched_image_context'] # multihead tensor [batch, class, num heads, dim per head]
                elif model.enable_multihead_radzero_forward_pass:
                    results = model.multihead_radzero_classification_branch(
                        image_embeddings={
                            "image_cls_raw_features": [image_feat[:, 0, :] for image_feat in batch_image_features_raw],
                            "image_patch_raw_features": [image_feat[:, 1:, :] for image_feat in batch_image_features_raw]
                        },
                        label_embeddings={
                            "text_cls_raw_features": class_cls_features_raw,
                        },
                    )
                    t2i_preds = results['logits_label_to_image']
                    enriched_image_embeddings = results['radzero_enriched_image_context'] # multihead tensor [batch, class, num heads, dim per head]

                if self.dqn_similarity in ['BCEWithLogitsLoss', 'SigmoidFocalLoss', 'BCE_and_INFONCE', 'ASL', 'ASL_NO_CLIPPING', 'BCEWithLogitsLoss', 'BCEWithLogitsLoss_auxiliarySoftMax', 'BCEWithLogitsLoss_auxiliarySoftMax_imglabel_mpinfonce'] and self.dqn_is_bidirectional:
                    t2i_preds = torch.sigmoid(t2i_preds)
                    final_pred = t2i_preds
                elif self.dqn_similarity in ['BCEWithLogitsLoss', 'SigmoidFocalLoss', 'BCE_and_INFONCE', 'ASL', 'ASL_NO_CLIPPING', 'BCEWithLogitsLoss', 'BCEWithLogitsLoss_auxiliarySoftMax', 'BCEWithLogitsLoss_auxiliarySoftMax_imglabel_mpinfonce'] and not self.dqn_is_bidirectional:
                    t2i_preds = torch.sigmoid(t2i_preds)
                    final_pred = t2i_preds
                elif self.dqn_similarity == 'CrossEntropyLoss' and self.dqn_is_bidirectional:
                    # use similarity based metrics, check get_dqn_similarities from Carzero
                    i2t_preds = model.dqn(
                        [torch.concat([class_cls_features_raw.unsqueeze(1), class_label_patch_features_raw], dim=1)], 
                        batch_image_features_raw[:,0,:], 
                        is_train_mode=False
                    )
                    i2t_preds = torch.transpose(i2t_preds, 0, 1)
                    final_pred = (t2i_preds + i2t_preds) / 2
                    assert False, 'not support this at the moment'
                elif self.dqn_similarity == 'CrossEntropyLoss' and not self.dqn_is_bidirectional:
                    # use similarity based metrics, check get_dqn_similarities from Carzero
                    final_pred = t2i_preds

                predictions.append(final_pred.cpu().numpy())
                if enriched_image_embeddings is not None:
                    image_embeddings.append(enriched_image_embeddings.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)

            if len(image_embeddings) > 0:
                image_embeddings = np.concatenate(image_embeddings, axis=0)
        return predictions, image_embeddings

    def gather_zeroshot_dqn_classification_statistics(self, class_list, predictions, label_names):
        # gather statistics
        result, class_counts = {}, {}
        for i, class_name in enumerate(class_list):
            class_pred = predictions[:, i, :]
            y_true = [1 if disease_name_mapping(class_name) in disease_list_mapping(row_labels) else 0 for row_labels in label_names]
            assert len(y_true) == class_pred.shape[0], 'label size does not match the number of image instances.'

            result[class_name] = {}
            fpr, tpr, _ = metrics.roc_curve(y_true, class_pred) # get the positive class probability
            result[class_name]["AUROC"] = metrics.auc(fpr, tpr) # micro AUC for each class then average it to get MACRO auc.
            result[class_name]["PR_AUROC"] = metrics.average_precision_score(y_true, class_pred)
            result[class_name]["Accuracy"] = metrics.accuracy_score(y_true, (class_pred > 0.5).astype(int))
            result[class_name]["F1"] = metrics.f1_score(y_true, (class_pred > 0.5).astype(int))
            y_pred_bin = (class_pred > 0.5).astype(int)
            result[class_name]["Precision"] = metrics.precision_score(y_true, y_pred_bin, zero_division=0)
            result[class_name]["Recall"] = metrics.recall_score(y_true, y_pred_bin, zero_division=0)
            class_counts[class_name] = sum(y_true) # for weighted average
        return result, class_counts

    def gather_zeroshot_binary_classification_statistics(self, class_list, predictions, label_names):
        result = {}
        for class_name in class_list:
            y_true = [1 if disease_name_mapping(class_name) in disease_list_mapping(row_labels) else 0 for row_labels in label_names]
            assert len(y_true) == predictions.shape[0], 'label size does not match the number of image instances.'

            result[class_name] = {}
            fpr, tpr, _ = metrics.roc_curve(y_true, predictions[:, 1])
            result[class_name]["AUROC"] = metrics.auc(fpr, tpr) # micro AUC for each class then average it to get MACRO auc.
            result[class_name]["PR_AUROC"] = metrics.average_precision_score(y_true, predictions[:, 1])
            result[class_name]["Accuracy"] = metrics.accuracy_score(y_true, np.argmax(predictions, axis=1))
            result[class_name]["F1"] = metrics.f1_score(y_true, np.argmax(predictions, axis=1))
        return result

