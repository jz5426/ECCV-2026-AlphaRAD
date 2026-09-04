import ast
import json
import random
from typing import Dict, List
import logging
import numpy as np
import pandas as pd
import torch
from nltk import tokenize
from torch.utils.data.dataset import Dataset
from cxrclip.data.data_utils import load_transform, transform_image
from cxrclip.util.utils import map_column, anonymize_qwen30b_204disease_descriptions, curate_dqn_input_labels
from cxrclip.prompt import constants
import cv2

log = logging.getLogger(__name__)

class ImageTextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        name: str,
        data_path: str,
        split: str,
        normalize: str,
        use_random_description: bool = False,
        random_description_prob: float = 0.0,
        add_alignment_prompt_prefix: bool = False,
        text_max_length: int = 256,
        prompt_max_length: int = 97,
        dqn_label_max_length: int = 48,
        text_sampling: str = "random",
        observation_explanation: bool = None,
        anonymize_observation_explanation: bool = False,
        loss_config: Dict = None,
        transform_config: Dict = None,
        prompt_from_json: bool = False,
        augmentation_type: str = None,
        data_frac: float = 1.0,
        num_negs: int = 0,
        alignment_prompt_col: str = None,
        labels_sample_size: int = None,
        enable_positive_negative_label_swapping: bool = False,
        report_col: str = "text", # can be alignment prompt
        augmented_report_col: str = "text_augment",
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.text_max_length = text_max_length
        self.prompt_max_length = prompt_max_length
        self.dqn_label_max_length = dqn_label_max_length
        self.text_sampling = text_sampling
        self.data_frac = data_frac
        self.num_negs = num_negs
        self.normalize = normalize

        self.chexpert5x200_disease_list = ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural effusion"]
        self.default_disease_list = [
            # unique for nih14
            "atelectasis",
            "cardiomegaly",
            "consolidation",
            "edema",
            "effusion",
            "emphysema",
            "fibrosis",
            "hernia",
            "infiltration",
            "mass",
            "nodule",
            "pleural thickening",
            "pneumonia",
            "pneumothorax",
            # unique for vindr
            "aortic enlargement",
            "calcification",
            "interstitial lung disease",
            "lung opacity",
            "mediastinal shift",
            "pleural effusion",
            "pulmonary fibrosis",
            "rib fracture",
            "lung tumor",
            "tuberculosis",
            "lung cavity",
            "lung cyst",
            # unique for chexpert
            "enlarged cardiomediastinum",
            "no findings" # NOTE: this is optional, may have better performance with this.
            # "lung lesion", # not exists in the csv file
        ]

        # use for report selection
        self.report_col = report_col
        self.augmented_report_col = augmented_report_col

        self.tokenizer = tokenizer

        assert hasattr(self.tokenizer, 'dual_cls')

        self.image_transforms = load_transform(split=split, transform_config=transform_config)
        if prompt_from_json:
            with open("datasets/train_prompts_all.json") as f:
                self.prompt_json = json.load(f)
        else:
            self.prompt_json = False

        assert data_path.endswith(".csv")

        self.df = pd.read_csv(data_path)
        # Remove rows where 'no findings' appears in the text_labels list
        # self.df = self.df[~self.df['text_labels'].apply(lambda x: 'no findings' in ast.literal_eval(x))].reset_index(drop=True)

        if "multihot204_qwen30b" in data_path:
            # the labels need to be extracted from a seperate file most likely
            self.pos_label_col = kwargs.get('positive_label_col')
            self.neg_label_col = kwargs.get('negative_label_col')
            # Convert stringified lists → Python lists
            positives_list = self.df[self.pos_label_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            negatives_list = self.df[self.neg_label_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # Flatten and combine
            all_positives = [item for sublist in positives_list for item in sublist]
            all_negatives = [item for sublist in negatives_list for item in sublist]
            self.world_labels = sorted(list(set(all_positives) | set(all_negatives))) # reference of the order of the labels

        elif 'multihot204' in data_path:
            self.text_labels = [ast.literal_eval(tl) for tl in self.df['text_labels']]
            start_idx = self.df.columns.get_loc('central venous catheter via subclavian vein')
            self.multihotlabels = self.df.iloc[:, start_idx:] # get the all of the labels of interest.
            assert self.multihotlabels.shape[1] == 204, "number of labels do not match what is indicated."

        self.add_alignment_prompt_prefix = add_alignment_prompt_prefix # whether add "There is"
        self.label_prompt_template = "There is {}." if self.add_alignment_prompt_prefix else "{}."

        if self.add_alignment_prompt_prefix:
            self.positive_prompt_template = "There is {}."
            self.negative_prompt_template = "There is no {}."
        else:
            self.positive_prompt_template = "{}."
            self.negative_prompt_template = "No {}."

        self.observation_explanation = dict({})
        if observation_explanation is not None:
            log.info('[ImageTextDataset]: using observation explanation for training.')
            with open(f"/cluster/projects/mcintoshgroup/CXR-CLIP/mimic-cxr-reports/{observation_explanation}") as f:
                self.observation_explanation = json.load(f)

            updated_oe = {}
            for key, value in self.observation_explanation.items():
                if isinstance(value, str):
                    updated_oe[key.lower()] = [value.lower()]
                elif isinstance(value, dict):
                    updated_oe[key.lower()] = [desp.lower() for desp in value['descriptions']]
            self.observation_explanation = updated_oe
            self.observation_explanation = self.observation_explanation | constants.covid_description
            missing_count = sum(k not in self.observation_explanation for k in self.world_labels)
            log.info(f'[ImageTextDataset]: There are labels with missing descriptions {missing_count}.')
            # command to check whether the diseases are loaded correctly
            # [k for k in self.default_disease_list if k not in self.observation_explanation]
            # [k for k in constants.PHYSICIAN_PADCHEST207 if k not in self.observation_explanation]
            if 'locationless' in observation_explanation:
                assert missing_count <= 3534, "Wrong combination of dataset train and dataset descriptions"
            else:
                assert missing_count <= 10975, "Wrong combination of dataset train and dataset descriptions"

        self.use_random_description = use_random_description
        self.random_description_prob = random_description_prob
        log.info(f'[ImageTextDataset]: using random description is {use_random_description}')
        self.anonymize_observation_explanation = anonymize_observation_explanation
        log.info(f'[ImageTextDataset]: using anonymized description is {anonymize_observation_explanation}')

        # only valid for the qwen30b_204disease_descriptions.json file (hand curated)
        if len(self.observation_explanation) > 0 and self.anonymize_observation_explanation and observation_explanation == 'qwen30b_204disease_descriptions.json':
            self.observation_explanation = anonymize_qwen30b_204disease_descriptions(self.observation_explanation)

        # sampple a subset of label columns for DQN learning.
        self.labels_sample_size = labels_sample_size
        self.enable_positive_negative_label_swapping = enable_positive_negative_label_swapping

        if data_frac < 1.0:
            self.df = self.df.sample(frac=self.data_frac, random_state=1, ignore_index=True)

        self.loss_config = {k: v for k, v in loss_config.items()}
        self.augmentation_type = augmentation_type
        self.alignment_prompt_col = alignment_prompt_col
        self.positive_label_noise_prob = kwargs.get("positive_label_noise_prob", 0.0)
        self.negative_label_noise_prob = kwargs.get("negative_label_noise_prob", 0.0)
        assert not (self.positive_label_noise_prob > 0.0 and self.negative_label_noise_prob > 0.0), \
            "Only one of positive_label_noise_prob or negative_label_noise_prob can be non-zero at a time."
        assert self.positive_label_noise_prob <= 1 and self.negative_label_noise_prob <= 1, \
            "Invalid probability for label noise."

        if self.augmentation_type in ['cxrclip', 'two_images_two_texts_one_prompt']:
            self.image_view_aug = True
            self.image_aug_other_image = True
            self.has_backtranslated = hasattr(self.df, self.augmented_report_col)

        self.image_aug_transforms = self.image_transforms

    def __len__(self):
        return len(self.df)

    ### CXRCLIP STYLE 
    def _cxrclip_get_images(self, row_index):
        if hasattr(self.df, "AP"):  # AP / PA / Lateral

            # get a list of views available for this study 
            try:
                view_list = ast.literal_eval(self.df["view"][row_index])
            except Exception:
                view_list = [self.df["view"][row_index]]

            # map the view tag to one of [AP, PA, LATERAL]
            view_list = list(map(map_column, view_list))

            # there are more than one type of views for this study
            if len(view_list) > 2:
                view_list = np.random.choice(view_list, size=2, replace=False)
                image_path_list = []
                for view in view_list:
                    try:
                        image_path_list = ast.literal_eval(self.df[view][row_index])
                    except Exception:
                        image_path_list = [self.df[view][row_index]]

                    image_path = np.random.choice(image_path_list, size=1)[0]
                    image_path_list.append(image_path)

            # when only has one type of view for this study
            else:
                if len(view_list) == 1: 
                    tag = view_list[0] # view column value
                else:
                    tag = "image"
    
                try:
                    image_path_list = ast.literal_eval(self.df[tag][row_index])
                except Exception:
                    image_path_list = [self.df[tag][row_index]]

                if self.split == "train":
                    if self.image_aug_other_image and len(image_path_list) > 1:
                        image_path_list = np.random.choice(image_path_list, size=2, replace=False)
                    else:
                        image_path_list = np.random.choice(image_path_list, size=1)
        else:
            # when the view is not indicated, then just use the image from the "image" column
            try:
                image_path_list = ast.literal_eval(self.df["image"][row_index])
            except Exception:
                image_path_list = [self.df["image"][row_index]]

        # pick the first image as the original image and data augmentation
        # image_original = Image.open(image_path_list[0]).convert("RGB")
        image_original = cv2.imread(image_path_list[0], cv2.IMREAD_GRAYSCALE)
        image_original = np.stack([image_original] * 3, axis=-1)
        image = transform_image(self.image_transforms, image_original, normalize=self.normalize)

        if self.image_view_aug:
            # in here, line 105 already selected two images.
            if len(image_path_list) > 1:
                # image_original = Image.open(image_path_list[1]).convert("RGB")
                # image_original = cv2.imread(image_path_list[1], cv2.IMREAD_GRAYSCALE)
                image_original = cv2.imread(np.random.choice(image_path_list[1:]), cv2.IMREAD_GRAYSCALE)
                image_original = np.stack([image_original] * 3, axis=-1)
            image_view = transform_image(self.image_aug_transforms, image_original, normalize=self.normalize)

        return image, image_view

    def _cxrclip_get_texts(self, row_index):
        # Get Text or Prompt
        if hasattr(self.df, self.report_col):
            try:
                text_list = ast.literal_eval(self.df[self.report_col][row_index])
            except Exception:
                # more likely the input is "text", which give exception (instead of "'text'") => wrap it as a list to backward compatible
                text_list = [self.df[self.report_col][row_index]]

            if self.has_backtranslated:
                try:
                    text_aug_list = ast.literal_eval(self.df[self.augmented_report_col][row_index])
                except Exception:
                    # more likely the input is "text", which give exception (instead of "'text'") => wrap it as a list to backward compatible
                    text_aug_list = [self.df[self.augmented_report_col][row_index]]

            if len(text_list) >= 2:
                indexes = np.random.randint(len(text_list), size=2)  # Multiple section
                text = text_aug_list[indexes[0]] if random.random() < 0.5 and self.has_backtranslated else text_list[indexes[0]]
                text2 = text_aug_list[indexes[1]] if random.random() < 0.5 and self.has_backtranslated else text_list[indexes[1]]

            else:
                if random.random() < 0.5:
                    text = text_list[0]
                    text2 = text_aug_list[0] if self.has_backtranslated else text_list[0]
                else:
                    text = text_aug_list[0] if self.has_backtranslated else text_list[0]
                    text2 = text_list[0]

            if self.split == "train":  # Text shuffle augment; invariant to the sentence ordering
                text_list = tokenize.sent_tokenize(text, language="english")
                random.shuffle(text_list)
                text = " ".join(text_list)

                text2_list = tokenize.sent_tokenize(text2, language="english")
                random.shuffle(text2_list)
                text2 = " ".join(text2_list)

        else:
            raise AttributeError("There is no report column in DataFrame.")
        
        return text, text2

    def get_two_images_two_texts(self, row_index):
        image, image_view = self._cxrclip_get_images(row_index)
        text, text2 = self._cxrclip_get_texts(row_index)

        out = {"image": image, "image_view": image_view, "text": text, "text2": text2}
        return out
    
    def _default_get_image(self, row_index):
        """no augmentation at all, just select the AP image if there is, otherwise just select a random image from an study"""
        image_path = None

        # get a list of possible views for this study 
        try:
            view_list = ast.literal_eval(self.df["view"][row_index])
        except Exception:
            view_list = [self.df["view"][row_index]]

        # view prioritization
        tag = None
        if "AP" in view_list:
            tag = "AP"
        elif "PA" in view_list:
            tag = "PA"
        elif "Lateral" in view_list:
            tag = "Lateral"
        
        # if image view is tagged.
        if tag is not None:
            try:
                image_path = ast.literal_eval(self.df[tag][row_index])[0]
            except Exception:
                image_path = self.df[tag][row_index]

        # when the image view is not tagged, randomly select one.
        if not image_path:
            try:
                image_path_list = ast.literal_eval(self.df["image"][row_index])
            except Exception:
                image_path_list = [self.df["image"][row_index]]
            image_path = np.random.choice(image_path_list, size=1)[0]

        # augment the single image and return
        # image_original = Image.open(image_path).convert("RGB")
        image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_original = np.stack([image_original] * 3, axis=-1)
        image = transform_image(self.image_transforms, image_original, normalize=self.normalize)
        return image
    
    def _default_get_text(self, row_index):
        """no augmentation at all, just select text from the self.report_col"""
        # use the non-augmented text aka: the clean text
        if hasattr(self.df, self.report_col):
            try:
                text = " ".join(ast.literal_eval(self.df[self.report_col][row_index])).strip()
            except Exception:
                text = self.df[self.report_col][row_index].strip()
        else:
            raise AttributeError("There is no report column in DataFrame.")

        if self.split == "train":  # Text shuffle augment; invariant to the sentence ordering
            _text_list = tokenize.sent_tokenize(text, language="english")
            random.shuffle(_text_list)
            sent_shuffled_text = " ".join(_text_list)
            return sent_shuffled_text

        # non training mode
        return text

    def get_one_image_one_text(self, row_index):
        """contrastive learning in pure CLIP style"""
        image = self._default_get_image(row_index)
        text = self._default_get_text(row_index)

        out = {"image": image, "text": text, "index": row_index}
        return out

    def _get_prompt(self, row_index, prompt_concatenation=False):
        if hasattr(self.df, self.alignment_prompt_col):
            try:
                prompts = ast.literal_eval(self.df[self.alignment_prompt_col][row_index])
            except Exception:
                prompts = [self.df[self.alignment_prompt_col][row_index]]

            if prompt_concatenation:
                # choose a random k between 1 and N
                k = np.random.randint(1, len(prompts)+1)

                # pick k unique items
                selected_prompts = np.random.choice(prompts, size=k, replace=False)
                np.random.shuffle(selected_prompts)  # in-place shuffle
                selected_prompt = " ".join(s.strip() for s in selected_prompts).strip()
            else:
                # randomly select one alignment prompt
                selected_prompt = np.random.choice(prompts, size=1)[0].strip()
        else:
            raise AttributeError("There is no alignment prompt column in DataFrame.")
        return selected_prompt
    
    def get_two_images_one_text_one_prompt(self, row_index):
        """for filip style training."""
        image, image_view = self._cxrclip_get_images(row_index)
        text, prompt = self._default_get_text(row_index), self._get_prompt(row_index)
        out = {"image": image, "image_view": image_view, 'text': text, 'prompt': prompt, "index": row_index}
        return out

    def get_one_image_one_text_one_prompt_with_labels(self, row_index):

        results = self.get_one_image_one_text_one_prompt(row_index)
        label_results = self._select_per_instance_diseases(row_index)
        
        return results | label_results
    
    def get_one_image_concatenated_aligment_prompts_with_labels(self, row_index):
        """
        - one image
        - concatenated clean prompts based on the labels
        - the labels
        """
        label_results = self._select_per_instance_diseases(row_index)

        # all the labels: label_results['selected_all_diseases_names']
        # all the positive labels: label_results['positive_disease_names']
        return

    def get_one_image_one_text_with_labels(self, row_index):

        results = self.get_one_image_one_text(row_index)
        label_results = self._select_per_instance_diseases(row_index)
        return results | label_results

    def _select_per_instance_diseases(self, row_index):
        """
        helper function to get labels
        """
        positive_disease_labels = self.text_labels[row_index]
        multihot = self.multihotlabels.iloc[row_index]
        if isinstance(self.labels_sample_size, int) and self.labels_sample_size > 0:
            ##: make sure all the positive labels inside and randomly sample the rest.

            # Step 1: Compute label frequencies (proportion of 1s per column)
            label_probabilities = self.multihotlabels.mean()

            # Step 2: Filter out the default diseases
            # candidate_labels = list(set(self.multihotlabels.columns) - set(self.default_disease_list))
            candidate_labels = list(set(self.default_disease_list) - set(positive_disease_labels))
            candidate_probabilities = label_probabilities[candidate_labels] + 1e-8

            # Step 3: Normalize the probabilities to sum to 1
            # candidate_probabilities = np.maximum(candidate_probabilities.values, 1e-8)
            probabilities = candidate_probabilities / candidate_probabilities.sum()

            # Step 4: Sample based on label frequency
            sampled_labels = np.random.choice(
                candidate_labels,
                size=min(self.labels_sample_size - len(positive_disease_labels), len(candidate_labels)),
                replace=False,
                p=probabilities.values
            )

            # Combine text labels with sampled labels, can be negative or positives (ensured distinct)
            # selected_all_diseases = list(set(self.default_disease_list).union(sampled_labels))
            selected_all_diseases = list(set(positive_disease_labels).union(sampled_labels))

        elif isinstance(self.labels_sample_size, str) and self.labels_sample_size == 'vindr_nih14_chexpert':
            selected_all_diseases = self.default_disease_list
        elif isinstance(self.labels_sample_size, str) and self.labels_sample_size == 'chexpert_5x200':
            selected_all_diseases = self.chexpert5x200_disease_list
        elif isinstance(self.labels_sample_size, str) and self.labels_sample_size == 'all_combined':
            selected_all_diseases = self.multihotlabels.columns.to_list() # all the combined diseases
        else:
            assert False, "Unrecognized labels_sample_size"

        assert multihot.loc[positive_disease_labels].sum() == len(positive_disease_labels), "Integrity constraint violation issue."
        return {
            # NOTE: only the following is used for the CE loss, the selected_all_diseases will contains all diseases of interest, 
            # containing ground truth 0 or 1
            "multihot_vector": multihot.loc[selected_all_diseases].to_numpy(dtype=int), # contains all the positives for this instance
            # the following only kept for reference only
            "positive_disease_names": positive_disease_labels,
            "selected_all_diseases_names": selected_all_diseases
        }

    def _negate_labels(self, label_list):
        negated_labels = []

        for label in label_list:
            if 'no findings' in label.lower():
                negated_labels.append(label.lower())
            elif 'no ' in label.strip().lower()[: len('no ')]:
                label = label.replace('no ', '').strip().lower()
                negated_labels.append(label)
            else:
                negated_labels.append('no '+label.strip().lower())
        return negated_labels

    def _randomly_negate_labels(self, label_list):
        original_labels = []
        negated_labels = []

        for label in label_list:
            label_clean = label.strip().lower()

            # Always keep "no findings" unchanged
            if 'no findings' in label_clean:
                original_labels.append(label_clean)
                continue

            # 50/50 decision
            if random.random() < 0.5:
                # keep as-is
                original_labels.append(label_clean)
            else:
                # negate
                if label_clean.startswith('no '):
                    neg_label = label_clean[len('no '):].strip()
                else:
                    neg_label = 'no ' + label_clean

                negated_labels.append(neg_label)

        return original_labels, negated_labels
    
    def get_one_image_one_text_worldlabels(self, row_index):
        """
        - select one image and one text
        - select the positive labels(mandatory) and randomly a select the negative labels
        """
        # select a single image follow the default
        results = self.get_one_image_one_text(row_index)
        pos_labels = ast.literal_eval(self.df[self.pos_label_col][row_index])
        neg_labels = ast.literal_eval(self.df[self.neg_label_col][row_index])

        x = random.choice([0, 1])
        if self.enable_positive_negative_label_swapping and len(neg_labels) > 0 and x == 1:
            new_neg_labels, new_pos_labels = self._negate_labels(pos_labels), self._negate_labels(neg_labels)
            neg_labels = new_neg_labels
            pos_labels = new_pos_labels

            # 'no findings' label is always positive label, never negative if exists
            NO_FINDING_STR = 'no findings'
            if NO_FINDING_STR in neg_labels:
                pos_labels.append(NO_FINDING_STR)
                neg_labels.pop(neg_labels.index(NO_FINDING_STR))

        return results | {
            "positive_disease_names": pos_labels,
            "negative_disease_names": neg_labels
        }

    def get_one_image_worldlabels(self, row_index):
        """
        - select one image
        - select the positive labels(mandatory) and randomly a select the negative labels
        """
        # select a single image follow the default
        image = self._default_get_image(row_index)
        pos_labels = ast.literal_eval(self.df[self.pos_label_col][row_index])
        neg_labels = ast.literal_eval(self.df[self.neg_label_col][row_index])

        # compute the multi-hot vector in the collate function
        return {
            "image": image,
            "index": row_index,
            "positive_disease_names": pos_labels,
            "negative_disease_names": neg_labels
        }

    def get_one_image_one_concatenatedPrompt_worldlabels(self, row_index):
        """
        - select one image and one text
        - select the positive labels(mandatory) and randomly a select the negative labels
        """
        # select a single image follow the default
        image = self._default_get_image(row_index)
        pos_labels = ast.literal_eval(self.df[self.pos_label_col][row_index])
        neg_labels = ast.literal_eval(self.df[self.neg_label_col][row_index])

        # get random concatenated prompt (positive and negatives)
        pos_prompts = [self.positive_prompt_template.format(pos) for pos in pos_labels]
        neg_prompts = [self.negative_prompt_template.format(neg) for neg in neg_labels]

        # combine
        all_prompts = pos_prompts + neg_prompts

        # shuffle in-place to make the model prompt order invariant.
        random.shuffle(all_prompts)

        # join into a single string
        text = " ".join(all_prompts)

        # compute the multi-hot vector in the collate function
        return {
            "image": image,
            "index": row_index,
            "text": text,
            "positive_disease_names": pos_labels,
            "negative_disease_names": neg_labels
        }

    def get_one_image_one_text_one_prompt(self, row_index):
        """
        use for forwarding the text report and prompt to the text encoder and then use dual embedding alignment
        """
        # select a single image follow the default
        image = self._default_get_image(row_index)

        # select the report and prompt
        text, prompt = self._default_get_text(row_index), self._get_prompt(row_index)
        out = {"image": image, 'text': text, 'prompt': prompt, "index": row_index}

        return out

    def get_one_image_one_text_random_prompts(self, row_index):
        # select a single image follow the default
        image = self._default_get_image(row_index)

        # select the report and prompt
        text, prompt = self._default_get_text(row_index), self._get_prompt(row_index, prompt_concatenation=True)
        out = {"image": image, 'text': text, 'prompt': prompt, "index": row_index}

        return out
    
    def _entity_augmentation(self, labels_list):
        """
            - there could the be the following three cases
            1. keep only the descriptions from self.observation_expalantion, that's it.
            2. keep both the entity (with prompt template if configured) and descriptions, that's it.
            3. keep only the entity (with prompt template if configured) and that's it
        """
        label_descriptions = []

        for pos in labels_list:
            descriptions = self.observation_explanation.get(pos, [])

            if len(descriptions) > 0:
                if pos == 'no findings':
                    description = ''
                else:
                    assert len(descriptions) > 0, f'Each disease should has a description but {pos} does not have.'
                    description = descriptions[random.randint(0, len(descriptions)-1)] if self.use_random_description else descriptions[0]

                x = np.random.choice([1, 2], p=[self.random_description_prob, 1-self.random_description_prob])
                if x == 1: # keep only the description of the entity
                    augmented_entity = self.label_prompt_template.format(description.lower()) # newly modified.
                elif x == 2: # keep only the entity names (most frequently use)
                    augmented_entity = self.label_prompt_template.format(pos)
            else:
                # only keep the disease name and no descriptions
                augmented_entity = self.label_prompt_template.format(pos)

            label_descriptions.append(augmented_entity.strip())
        
        return label_descriptions

    def get_two_images_two_texts_one_prompt(self, row_index):
        """
        used for the following:
            - dual image views and texts for the same cls token like CXRCLIP
            - the prompt is used for aligning the image views only without the text-side augmentation from CXRCLIP
        """
        # select the image follow the cxrclip
        image, image_view = self._cxrclip_get_images(row_index)

        # select the report and prompt
        text, text2 = self._cxrclip_get_texts(row_index)
        prompt = self._get_prompt(row_index)

        out = {"image": image, "image2": image_view, 'text': text, 'text2': text2, 'prompt': prompt, "index": row_index}
        return out

    def __getitem__(self, index):

        # use different kind of augmentation here, or no augmentation as default

        if self.augmentation_type == 'cxrclip':
            return self.get_two_images_two_texts(index)
        elif self.augmentation_type == 'one_image_one_text_one_prompt':
            return self.get_one_image_one_text_one_prompt(index)
        elif self.augmentation_type == 'one_image_one_text_one_prompt_with_labels':
            return self.get_one_image_one_text_one_prompt_with_labels(index)
        elif self.augmentation_type == 'one_image_one_text_random_prompts':
            return self.get_one_image_one_text_random_prompts(index) # filip style
        elif self.augmentation_type == 'two_images_one_text_one_prompt':
            return self.get_two_images_one_text_one_prompt(index)
        elif self.augmentation_type == 'two_images_two_texts_one_prompt':
            return self.get_two_images_two_texts_one_prompt(index)
        elif self.augmentation_type == 'clip':
            return self.get_one_image_one_text(index) # default option like vanilla CLIP
        elif self.augmentation_type == 'clip_with_labels':
            return self.get_one_image_one_text_with_labels(index)
        elif self.augmentation_type == 'clip_with_shared_labels':
            return self.get_one_image_one_text_with_labels(index)
        elif self.augmentation_type in ['clip_with_shared_worldLabels', 'clip_with_diseaseEntityAugmentation']:
            return self.get_one_image_one_text_worldlabels(index)
        elif self.augmentation_type in ['image_shared_worldlabels']:
            return self.get_one_image_worldlabels(index)
        elif self.augmentation_type == 'one_image_one_concatenatedPrompt_shared_worldLabels':
            return self.get_one_image_one_concatenatedPrompt_worldlabels(index)
        elif self.augmentation_type == 'one_image_concatenated_prompts_with_labels':
            return self.get_one_image_concatenated_aligment_prompts_with_labels(index)
        assert False, 'Unrecognized augmentation type'


    def collate_fn(self, instances: List):
        """batchify the stuffs"""

        if self.augmentation_type == 'cxrclip':
            images = torch.stack([ins["image"] for ins in instances], dim=0)
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", 
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            texts2 = list([ins["text2"] for ins in instances])
            text_tokens2 = self.tokenizer(texts2, padding="max_length", truncation=True, return_tensors="pt", 
                                          max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)
            images2 = torch.stack([ins["image_view"] for ins in instances], dim=0)

            batch = {
                "images": images,
                "image_views": images2,
                "texts": texts,
                "texts2": texts2,
                "text_tokens": text_tokens,
                "text_tokens2": text_tokens2,
            }
        elif (self.augmentation_type == 'one_image_one_text_one_prompt' or
              self.augmentation_type == 'one_image_one_text_random_prompts'
            ):
            images = torch.stack([ins["image"] for ins in instances], dim=0)
            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         return_offsets_mapping=True,
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # need token at index 1 of in hidden_state
            prompts = list([ins['prompt'] for ins in instances])
            prompt_tokens = self.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt",
                                           return_offsets_mapping=True,
                                           max_length=self.prompt_max_length + 1 if self.tokenizer.dual_cls else self.prompt_max_length)

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "text_offset_mapping": text_tokens.pop('offset_mapping'),
                "prompts": prompts,
                "prompt_tokens": prompt_tokens,
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                "prompt_offset_mapping": prompt_tokens.pop('offset_mapping')
            }

        elif self.augmentation_type == 'one_image_one_text_one_prompt_with_labels':
            images = torch.stack([ins["image"] for ins in instances], dim=0)
            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # need token at index 1 of in hidden_state
            prompts = list([ins['prompt'] for ins in instances])
            prompt_tokens = self.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt",
                                           max_length=self.prompt_max_length + 1 if self.tokenizer.dual_cls else self.prompt_max_length)
            
            # this assume each instance have different set of label samples.
            labels = list([ins['selected_all_diseases_names'] for ins in instances])
            label_tokens = []
            for instance_labels in labels:
                label_tokens = self.tokenizer(
                    [self.label_prompt_template.format(label) for label in instance_labels], 
                    padding="max_length", truncation=True, return_tensors="pt",
                    max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
                )
                label_tokens.append(label_tokens) # [number of sample disease label for that particular instance, text length]

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                # "text_offset_mapping": text_tokens.pop('offset_mapping'),
                "prompts": prompts,
                "prompt_tokens": prompt_tokens,
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                # "prompt_offset_mapping": prompt_tokens.pop('offset_mapping'),
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": [ins['multihot_vector'] for ins in instances],
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": [ins['selected_all_diseases_names'] for ins in instances]
            }
        elif self.augmentation_type == 'two_images_one_text_one_prompt':

            images = torch.stack([ins["image"] for ins in instances], dim=0)
            images2 = torch.stack([ins["image_view"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # need token at index 1 of in hidden_state
            prompts = list([ins['prompt'] for ins in instances])
            prompt_tokens = self.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt",
                                           max_length=self.prompt_max_length + 1 if self.tokenizer.dual_cls else self.prompt_max_length)

            batch = {
                "images": images,
                "image_views": images2,
                "texts": texts,
                "text_tokens": text_tokens,
                "prompts": prompts,
                "prompt_tokens": prompt_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "prompt_attention_mask": prompt_tokens["attention_mask"]
            }
        elif self.augmentation_type == 'two_images_two_texts_one_prompt':

            images = torch.stack([ins["image"] for ins in instances], dim=0)
            images2 = torch.stack([ins["image2"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # need token at index 0 of in hidden_state
            texts2 = list([ins["text2"] for ins in instances])
            text_tokens2 = self.tokenizer(texts2, padding="max_length", truncation=True, return_tensors="pt",
                                          max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # need token at index 1 of in hidden_state
            prompts = list([ins['prompt'] for ins in instances])
            prompt_tokens = self.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt",
                                           max_length=self.prompt_max_length + 1 if self.tokenizer.dual_cls else self.prompt_max_length)
            
            batch = {
                "images": images,
                "images2": images2,
                "texts": texts,
                "texts2": texts2,
                "text_tokens": text_tokens,
                "text_tokens2": text_tokens2,
                "prompts": prompts,
                "prompt_tokens": prompt_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "text2_attention_mask": text_tokens2["attention_mask"],
                "prompt_attention_mask": prompt_tokens["attention_mask"]
            }

        elif self.augmentation_type == 'clip':
            images = torch.stack([ins["image"] for ins in instances], dim=0)
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"]
            }
        elif self.augmentation_type == 'clip_with_labels':
            images = torch.stack([ins["image"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)

            # this assume each instance have different set of label samples.
            labels = list([ins['selected_all_diseases_names'] for ins in instances])
            label_tokens = []
            for instance_labels in labels:
                final_labels = []
                for l in instance_labels:
                    label = curate_dqn_input_labels(
                        l, 
                        self.label_prompt_template, 
                        self.observation_explanation, 
                        self.anonymize_observation_explanation,
                        random_description_selection=self.use_random_description
                    )
                    final_labels.append(label)

                label_tokens = self.tokenizer(
                    final_labels, 
                    padding="max_length", truncation=True, return_tensors="pt",
                    max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
                )
                label_tokens.append(label_tokens) # [number of sample disease label for that particular instance, text length]

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": [ins['multihot_vector'] for ins in instances],
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": [ins['selected_all_diseases_names'] for ins in instances]
            }
        elif self.augmentation_type == 'clip_with_shared_labels':
            images = torch.stack([ins["image"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)
            # construct shared text labels            
            labels = instances[0]['selected_all_diseases_names'] # shared labels across each batch
            final_labels = []

            # add disease description and prompt template if applicable if applicable.
            for l in labels:
                label = curate_dqn_input_labels(
                    l, 
                    self.label_prompt_template, 
                    self.observation_explanation, 
                    self.anonymize_observation_explanation,
                    random_description_selection=self.use_random_description
                )
                final_labels.append(label)

            # tokenize the disease description
            label_tokens = self.tokenizer(
                final_labels, 
                padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
            )

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": [ins['multihot_vector'] for ins in instances], # each instance in a batch contains different set of positive labels
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": self.multihotlabels.columns.to_list()
            }
        elif self.augmentation_type == 'clip_with_diseaseEntityAugmentation':
            images = torch.stack([ins["image"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state
            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)
            
            # accumulate all the positive_disease_names for each instance in instances
            all_pos = []
            for inst in instances:
                all_pos.extend(inst['positive_disease_names'])

            # 2) unique set of positive labels for this batch
            unique_pos = sorted(set(all_pos))  # sorted helps reproducibility

            # 3) randomly sample a set of negative labels (size 100)
            #    from world_labels excluding the already positive labels
            available_negs = list(set(self.world_labels) - set(unique_pos))
            remain_slots_to_sample = max(0, self.labels_sample_size - len(unique_pos)) # total of 300
            shared_sampled_neg = random.sample(available_negs, min(remain_slots_to_sample, len(available_negs)))

            # 4) merged label list, then shuffle but the same batch share the same order of the labels
            labels = unique_pos + shared_sampled_neg
            random.shuffle(labels)

            # 5) build multi-hot label vector for each instance
            #    shape: [batch_size, len(shared_labels)]
            num_labels, multi_hot_list = len(labels), []
            label_index_map = {label: idx for idx, label in enumerate(labels)}
            for inst in instances:
                arr = np.zeros(num_labels, dtype=np.float32)
                for p in inst['positive_disease_names']:
                    idx = label_index_map.get(p)
                    arr[idx] = 1.0
                assert sum(arr) > 0, "no positive labels."
                multi_hot_list.append(arr)     # append numpy array for this instance

            # add disease description if applicable. NOTE: main novelty in the data side.
            final_labels = self._entity_augmentation(labels)

            # tokenize the disease description
            label_tokens = self.tokenizer(
                final_labels, 
                padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
            )

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": multi_hot_list, # each instance in a batch contains different set of positive labels, as a list of numpy array
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": labels
            }
        elif self.augmentation_type == 'image_shared_worldlabels':
            images = torch.stack([ins["image"] for ins in instances], dim=0)

            # accumulate all the positive_disease_names for each instance in instances
            all_pos = []
            for inst in instances:
                all_pos.extend(inst['positive_disease_names'])

            # 2) unique set of positive labels for this batch
            unique_pos = sorted(set(all_pos))  # sorted helps reproducibility

            # 3) randomly sample a set of negative labels (size 100)
            #    from world_labels excluding the already positive labels
            available_negs = list(set(self.world_labels) - set(unique_pos))
            remain_slots_to_sample = max(0, self.labels_sample_size - len(unique_pos)) # total of 300
            shared_sampled_neg = random.sample(available_negs, min(remain_slots_to_sample, len(available_negs)))

            # 4) merged label list, then shuffle but the same batch share the same order of the labels
            labels = unique_pos + shared_sampled_neg
            random.shuffle(labels)

            # 5) build multi-hot label vector for each instance
            #    shape: [batch_size, len(shared_labels)]
            num_labels, multi_hot_list = len(labels), []
            label_index_map = {label: idx for idx, label in enumerate(labels)}
            for inst in instances:
                arr = np.zeros(num_labels, dtype=np.float32)
                for p in inst['positive_disease_names']:
                    idx = label_index_map.get(p)
                    arr[idx] = 1.0
                assert sum(arr) > 0, "no positive labels."
                multi_hot_list.append(arr)     # append numpy array for this instance

            # add disease description if applicable.
            final_labels = []
            for l in labels:
                label = curate_dqn_input_labels(
                    l, 
                    self.label_prompt_template, 
                    self.observation_explanation, 
                    self.anonymize_observation_explanation,
                    random_description_selection=self.use_random_description
                )
                final_labels.append(label)

            # tokenize the disease description
            label_tokens = self.tokenizer(
                final_labels, 
                padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
            )

            batch = {
                "images": images,
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": multi_hot_list, # each instance in a batch contains different set of positive labels, as a list of numpy array
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": labels
            }
        elif self.augmentation_type in ['clip_with_shared_worldLabels', 'one_image_one_concatenatedPrompt_shared_worldLabels']:
            images = torch.stack([ins["image"] for ins in instances], dim=0)

            # need token at index 0 of in hidden_state

            texts = list([ins["text"] for ins in instances])
            text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                         max_length=self.text_max_length + 1 if self.tokenizer.dual_cls else self.text_max_length)
            
            # accumulate all the positive_disease_names for each instance in instances
            all_pos = []
            for inst in instances:
                all_pos.extend(inst['positive_disease_names'])

            # 2) unique set of positive labels for this batch
            unique_pos = sorted(set(all_pos))  # sorted helps reproducibility

            # 3) randomly sample a set of negative labels (size 100)
            #    from world_labels excluding the already positive labels
            available_negs = list(set(self.world_labels) - set(unique_pos))
            remain_slots_to_sample = max(0, self.labels_sample_size - len(unique_pos)) # total of 300
            shared_sampled_neg = random.sample(available_negs, min(remain_slots_to_sample, len(available_negs)))

            shared_sampled_pos = []
            if self.enable_positive_negative_label_swapping:
                shared_sampled_neg, shared_sampled_pos = self._randomly_negate_labels(shared_sampled_neg)

            # 4) merged label list, then shuffle but the same batch share the same order of the labels
            labels = unique_pos + shared_sampled_neg + shared_sampled_pos
            random.shuffle(labels)

            # 5) build multi-hot label vector for each instance
            #    shape: [batch_size, len(shared_labels)]
            num_labels, multi_hot_list = len(labels), []
            label_index_map = {label: idx for idx, label in enumerate(labels)}
            for inst in instances:
                arr = np.zeros(num_labels, dtype=np.float32)
                for p in inst['positive_disease_names'] + shared_sampled_pos:
                    idx = label_index_map.get(p)
                    arr[idx] = 1.0
                assert sum(arr) >= len(shared_sampled_pos), "no positive labels."
                multi_hot_list.append(arr)     # append numpy array for this instance

            # 6+7) label noise: flip positives→0 and/or negatives→1 with configured probs.
            #      At least one entry of each kind is guaranteed to survive per instance.
            if self.split == "train" and (self.positive_label_noise_prob + self.negative_label_noise_prob) > 0:
                # 6) positive-label noise: independently flip each positive entry to 0
                #    with probability positive_label_noise_prob (train split only).
                #    At least one positive is guaranteed to survive per instance.
                if self.positive_label_noise_prob > 0.0:
                    for arr in multi_hot_list: # each instance in the batch.
                        pos_indices = np.where(arr == 1.0)[0]  # original positives before any flipping

                        # bernouli flipping
                        for idx in pos_indices:
                            if random.random() < self.positive_label_noise_prob:
                                arr[idx] = 0.0
                        # if every positive was flipped, restore one at random
                        if arr.sum() == 0:
                            arr[np.random.choice(pos_indices)] = 1.0

                # 7) negative-label noise: independently flip each negative entry to 1
                #    with probability negative_label_noise_prob (train split only).
                #    At least one negative is guaranteed to survive per instance.
                if self.negative_label_noise_prob > 0.0:
                    for arr in multi_hot_list:
                        neg_indices = np.where(arr == 0.0)[0]  # original negatives before any flipping

                        # bernoulli flipping
                        flipped = []
                        for idx in neg_indices:
                            if random.random() < self.negative_label_noise_prob:
                                arr[idx] = 1.0
                                flipped.append(idx)
                        # if every negative was flipped, restore one at random
                        if len(flipped) == len(neg_indices) and len(neg_indices) > 0:
                            restore_idx = np.random.choice(flipped)
                            arr[restore_idx] = 0.0

            # add disease description if applicable.
            final_labels = []
            for l in labels:
                label = curate_dqn_input_labels(
                    l,
                    self.label_prompt_template,
                    self.observation_explanation,
                    self.anonymize_observation_explanation,
                    random_description_selection=self.use_random_description,
                    probability_for_selecting_description=self.random_description_prob
                )
                final_labels.append(label)

            # tokenize the disease description
            label_tokens = self.tokenizer(
                final_labels, 
                padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
            )

            batch = {
                "images": images,
                "texts": texts,
                "text_tokens": text_tokens,
                "text_attention_mask": text_tokens["attention_mask"],
                "labels": labels,
                "label_tokens": label_tokens,
                "multihot_label": multi_hot_list, # each instance in a batch contains different set of positive labels, as a list of numpy array
                "positive_disease_names": [ins['positive_disease_names'] for ins in instances],
                "selected_all_diseases_names": labels,
            }
        else:
            assert False, 'Unrecognized augmentation type'
        
        # accumulate sampled index
        batch = {
            'index': torch.tensor([ins['index'] for ins in instances]),
            **batch,
        }

        return batch
    
