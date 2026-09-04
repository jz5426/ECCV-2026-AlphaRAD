from typing import Dict, Union

import albumentations
import numpy as np
from PIL import Image
from torchvision import transforms
from cxrclip.model.modules.text_encoder import CustomBertTokenizer

def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, dual_cls=False, **kwargs):
    tokenizer = CustomBertTokenizer(cache_dir, dual_cls, **kwargs)
    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms


def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize):
    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image
            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if normalize == "imagenet" or  normalize == 'dinov2':
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    elif normalize == "xraydino":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5307, 0.5307, 0.5307], std=[0.2583, 0.2583, 0.2583])(image)
    else:
        raise KeyError(f"Not supported Normalize: {normalize}")

    return image
