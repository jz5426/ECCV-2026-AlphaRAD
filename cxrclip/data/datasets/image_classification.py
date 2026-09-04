import ast
from typing import Dict, List

import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from cxrclip.data.data_utils import load_transform, transform_image
from cxrclip.evaluator_utils import disease_list_mapping, disease_name_mapping, rle2mask
from cxrclip.prompt import constants
from cxrclip.util.utils import curate_dqn_input_labels
import logging
log = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        normalize: str,
        data_frac: float = 1.0,
        sample_shots_not_percentage: bool = False,
        transform_config: Dict = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.data_frac = data_frac
        self.normalize = normalize

        self.image_transforms = load_transform(split=split, transform_config=transform_config)
        self.df = pd.read_csv(data_path)

        # for few shot
        self.tokenizer = kwargs.get('tokenizer', None)
        self.dqn_label_max_length = kwargs.get('dqn_label_max_length', 48)
        self.prompt_template = "There is {}." if kwargs.get('add_alignment_prompt_prefix', False) else "{}."

        self.enable_dqn_fewshot = kwargs.get('enable_dqn_fewshot', False)
        self.for_fewshot_fintuning = kwargs.get('for_fintuning', False)
        if 'class' in self.df:
            self.df['class'] = self.df['class'].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else [x])
        if 'label' in self.df:
            self.df['label'] = self.df['label'].apply(lambda x: ast.literal_eval(x) if type(x) is str and x.startswith('[') else [float(x)])

        # NOTE: make sure the following only execute in finetuning: remove empty/nonrelevant rows but these are relevant for pure zero-shot
        if self.for_fewshot_fintuning:
            self.disease_of_interest_list = [d.lower() for d in getattr(constants, name.upper())]

            # NOTE: consider only enable this during finetuning instead of few-shot, which affect cxrlt-task2
            # keep only the labels of interest based on the constants file
            self.df['class'] = self.df['class'].apply(lambda class_list: [
                c.lower() for c in class_list if c.lower() in self.disease_of_interest_list
            ])
            self.df = self._create_multi_hot_label_vector(self.df).reset_index(drop=True)

        # finetuning
        if self.for_fewshot_fintuning and data_frac < 1.0:
            assert not sample_shots_not_percentage, "integrity constraints violation."

            # 1. Explode 'class' so rows with multiple classes appear once per class
            exploded_df = self.df.explode('class')

            # 2. Group by the individual class names and sample the indices
            #    (random_state ensures reproducibility)
            sampled_indices = exploded_df.groupby('class').sample(frac=data_frac, random_state=42).index
            
            # 3. Filter original dataframe using the unique sampled indices
            #    This automatically handles deduplication if a row was selected by multiple classes
            self.df = self.df.loc[sampled_indices.unique()].reset_index(drop=True)

            # 4. Print the number of instances for each class (debug purposes)
            log.info(f"[ImageClassificationDataset] Sampled {len(self.df)} rows ({data_frac*100}%). Class distribution:")
            log.info(f"[ImageClassificationDataset] {self.df.explode('class')['class'].value_counts()}")

        elif self.for_fewshot_fintuning and sample_shots_not_percentage and data_frac >= 1.0: # sample_shots_not_percentage mainly targeting train split
            num_shots = int(data_frac)
            # 1. Explode 'class' so rows with multiple classes appear once per class
            exploded_df = self.df.explode('class')

            # 2. Group by class and sample N shots.
            #    We use a lambda to check the size of the group. 
            #    If a class has < num_shots, we take all of them (min(len(x), num_shots)).
            #    group_keys=False prevents Pandas from creating a MultiIndex.
            sampled_indices = exploded_df.groupby('class', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), num_shots), random_state=42)
            ).index

            # 3. Filter original dataframe using the unique sampled indices
            #    (Union of indices: an image is kept if it serves as a shot for ANY of its labels)
            self.df = self.df.loc[sampled_indices.unique()].reset_index(drop=True)

            # 4. Print the number of instances for each class
            log.info(f"[ImageClassificationDataset] Sampled {len(self.df)} rows ({num_shots} shots/class). Class distribution:")
            log.info(f"[ImageClassificationDataset] {self.df.explode('class')['class'].value_counts()}")

        elif data_frac == 1.0: # Intentional to avoid confusion
            # zeroshot case
            pass
        # NOTE: modify the df so that it contains a subset of diseases (dynamic filtering instead of do that each time in a spreadsheet)
    
    def _create_multi_hot_label_vector(self, dataframe):
        """
        mainly to override the label vector
        """
        # 1. Get the unique set of disease names and sort them
        #    (Using a set comprehension over the lists is efficient here)
        # unique_classes = {cls for classes in dataframe['class'] for cls in classes}
        # assert len(self.disease_of_interest_list) == len(unique_classes), "something went wrong"
        unique_classes = self.disease_of_interest_list

        # 2. Map each class name to a specific index (0, 1, 2...)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        num_classes = len(unique_classes)

        # 3. Define a helper to convert a list of names into a list of 0s and 1s
        def create_multi_hot(class_list):
            label = [0] * num_classes
            for c in class_list:
                # Set the index to 1 if the class exists in the row
                label[class_to_idx[c]] = 1
            return label

        # 4. Replace dataframe['label'] with the new multi-hot lists
        dataframe['label'] = dataframe['class'].apply(create_multi_hot)
        return dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df["image"][index]
        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = np.stack([image] * 3, axis=-1)
        # image = transform_image(self.image_transforms, image, normalize=self.normalize)

        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([original_image] * 3, axis=-1)
        image = transform_image(self.image_transforms, image, normalize=self.normalize)

        label = None
        if "label" in self.df:
            label = self.df["label"][index]
            if type(label) is str:
                label = ast.literal_eval(label)
            # else:
            #     label = [label]
            # if self.name == "vindr_cxr":
            #     # pop the multi-hot labels, not the label name
            #     label.pop(-2)  # other lesion
            #     label.pop(-1)  # other disease
            label = torch.Tensor(label)
        # else:
        #     raise AttributeError("Cannot read the column for label")

        label_name = None
        if "class" in self.df:
            label_name = self.df["class"][index]
            # if label_name.startswith("["):
            #     label_name = ast.literal_eval(label_name)
            # else:
            #     label_name = [label_name]
            if isinstance(label_name, str):
                label_name = [label_name]
        # else:
        #     raise AttributeError("Cannot read the column for label_name")
        
        mask = None
        if "EncodedPixels" in self.df:
            # refers to segmentation_utils.py for reference implementation where multiple masks are combined and only evaluted once for each unique dicom_id
            encoded_pixels = ast.literal_eval(self.df["EncodedPixels"][index])
            height, width = original_image.shape
            mask = np.zeros([height, width])
            if encoded_pixels[0].strip() != "-1":  
                for encoded_label in encoded_pixels:
                    mask += rle2mask(encoded_label.strip(), width, height)
            mask = torch.LongTensor(mask > 0).unsqueeze(0)
            assert mask.squeeze().shape == (height, width), f'mask and image size mismatch. mask: {mask.squeeze().shape}, mine: {(height, width)}'

        # for pointing game.
        boxes = []
        if "boxes" in self.df:
            boxes = self.df["boxes"][index]
            boxes = ast.literal_eval(boxes)

        return {
            "image": image, 
            "label": label, 
            "label_name": label_name, 
            "boxes": boxes,
            "mask": mask,
            "image_path": self.df['image'][index]
        }

    def collate_fn(self, instances: List):

        labels, masks, label_names = None, [], []
        if len(instances) > 0 and instances[0]["label"] is not None:
            labels = torch.stack([ins["label"] for ins in instances], dim=0)
        if len(instances) > 0 and instances[0]["mask"] is not None:
            # masks = torch.stack([ins["mask"] for ins in instances], dim=0)
            masks = [ins["mask"] for ins in instances]

        if len(instances) > 0 and instances[0]["label_name"] is not None:
            label_names = list([ins["label_name"] for ins in instances])

        images = torch.stack([ins["image"] for ins in instances], dim=0)
        boxes = list([ins["boxes"] for ins in instances])
        image_paths = list([ins["image_path"] for ins in instances])

        # shared dataset label query prompt for few-shot learning
        label_tokens = None
        if self.enable_dqn_fewshot and self.for_fewshot_fintuning:
            labels_query = []
            for disease_name in self.disease_of_interest_list:
                l = curate_dqn_input_labels(disease_name_mapping(disease_name), self.prompt_template)
                labels_query.append(l)
            label_tokens = self.tokenizer(
                labels_query, 
                padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.dqn_label_max_length + 1 if self.tokenizer.dual_cls else self.dqn_label_max_length
            )

        return {
            "images": images, 
            "labels": labels, 
            "label_names": label_names,
            "label_tokens": label_tokens,
            "boxes": boxes,
            "masks": masks,
            "image_paths": image_paths,
            "multihot_label": labels.numpy() if labels is not None else labels
        }
