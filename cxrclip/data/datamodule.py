import logging
from typing import Dict
from torch.utils.data import DataLoader
from .data_utils import load_tokenizer
from .datasets import load_dataset
log = logging.getLogger(__name__)

class DataModule:
    def __init__(
        self,
        data_config: Dict,
        dataloader_config: Dict = None,
        tokenizer_config: Dict = None,
        loss_config: Dict = None,
        transform_config: Dict = None,
    ):
        self.data_config = data_config
        self.dataloader_config = dataloader_config
        self.tokenizer_config = tokenizer_config
        self.loss_config = loss_config
        self.tokenizer = load_tokenizer(**self.tokenizer_config) if self.tokenizer_config is not None else None
        self.datasets = {"test": []}
        self.train_loader = None
        self.valid_loader_dict = None
        self.test_loader = None

        for split in data_config:
            dataset_split_config = self.data_config[split]
            for name in dataset_split_config:
                dataset_config = dataset_split_config[name]
                # entry point for ImageTextDataset
                dataset = load_dataset(
                    split=split, 
                    tokenizer=self.tokenizer, 
                    transform_config=transform_config, 
                    loss_config=self.loss_config,
                    **dataset_config
                )
                self.datasets[split].append(dataset)

                log.info(f"Dataset loaded: {dataset_split_config[name]['name']} for {split}")

    def test_dataloader(self):
        assert self.dataloader_config is not None
        if self.test_loader is None:
            self.test_loader = {
                test_dataset.name: DataLoader(
                    test_dataset, collate_fn=getattr(test_dataset, "collate_fn", None), **self.dataloader_config["test"]
                )
                for test_dataset in self.datasets["test"]
            }
        return self.test_loader
