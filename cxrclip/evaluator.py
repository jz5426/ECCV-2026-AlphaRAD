import logging
from typing import Dict
import torch
from cxrclip.base_evaluator import BaseEvaluator
from cxrclip.data import DataModule
from cxrclip.model import build_model
from cxrclip.evaluator_utils import *

log = logging.getLogger(__name__)

class Evaluator(BaseEvaluator):
    def __init__(self, eval_config: Dict, ckpt_paths, split: str ='valid'):
        """expand the original implementation to handle validation on the valid split.
        
        all possible configs should retrieved from ckpt_paths if possible to ensure maximum compatibility with
        standlone test on data_test.
        """
        ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)

        # TODO: set the cache_dir for both the text encoder and the image encoder.
        ckpt['config']['model']['text_encoder']['cache_dir'] = 'cache_path/to/ sentence-transformers/all-mpnet-base-v2 or Laihaoran/BioClinicalMPBERT'
        ckpt['config']['model']['image_encoder']['cache_dir'] = 'cache_path/to/xraydinov2'

        super(Evaluator, self).__init__(
            ckpt["config"], 
            rank=None
        )

        self.split = split
        assert split in ['test']
        assert "test" in eval_config and "checkpoint" in eval_config["test"], "Evaluation needs model checkpoint."
        eval_data_config = {f"{split}": eval_config[f"data_{split}"]}

        for _split in eval_data_config:
            for _dataset in eval_data_config[_split]:
                eval_data_config[_split][_dataset]["normalize"] = self.config['image_model']['normalize']

        image_transform = eval_config.get('transform', None)
        if image_transform is None:
            image_transform = self.config["transform"]

        if split == "valid":
            self.datamodule = DataModule(
                data_config=eval_data_config,
                dataloader_config=eval_config["dataloader"],
                tokenizer_config=self.config["tokenizer"] if "tokenizer" in self.config else None,
                loss_config=self.config["loss"], # note that for imagetext_eval data_type, this is not needed, but it is required for valid split
                transform_config=image_transform,
            )
        else:
            self.datamodule = DataModule(
                data_config=eval_data_config,
                dataloader_config=eval_config["dataloader"], # batch size and etc from eval_clip.yaml
                tokenizer_config=self.config["tokenizer"] if "tokenizer" in self.config else None,
                transform_config=image_transform,
            )

        self.data_loader_dict = self.datamodule.test_dataloader() if split == "test" else self.datamodule.valid_dataloader()
        assert len(self.data_loader_dict) > 0

        # optional: save per-image segmentation data for model comparison
        self.seg_save_dir = eval_config.get('seg_save_dir', None)

        # load a brand new model and load it with pretrained weights later in evaluate_clip_online
        self.model = build_model(
            model_config=self.config["model"], 
            loss_config=self.config["loss"], 
            tokenizer=self.datamodule.tokenizer
        )
        self.model = self.model.to(self.device)

    def evaluate_clip_offline(self, checkpoint, test_dataset_name):
        """Evaluator for zero-shot/retrieval."""

        log.info(f"Load model {checkpoint} for evaluation.")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        print(f'missing: {missing}')
        print(f'unexpected: {unexpected}')
        self.model.eval()
        with torch.no_grad():
            results = self.on_the_fly_evaluation_one_data(test_dataset_name, self.similarity_type)
        return results
