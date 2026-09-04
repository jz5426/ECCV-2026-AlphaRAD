import logging
from typing import Dict
from scipy.special import softmax
from cxrclip.base_evaluator import BaseEvaluator
from cxrclip.evaluator_utils import *
from cxrclip.prompt import constants

log = logging.getLogger(__name__)

class OnlineEvaluator(BaseEvaluator):
    """
    for evaluation that are not reloading the weights, during training.
    """
    def __init__(self, config: Dict, data_loader_dict, model, datamodule, rank=None):
        super(OnlineEvaluator, self).__init__(config, rank)

        self.model = model
        self.data_loader_dict = data_loader_dict
        self.datamodule = datamodule

    def evaluate_classifier_online(self, test_dataset_name):
        self.model.eval()
        dataloader = self.data_loader_dict[test_dataset_name]

        preds, labels = [], []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type):
                    out = self.model(batch, device=self.device)
                    preds.append(torch.sigmoid(out["cls_pred"]).detach().cpu().numpy())
                    labels.append(batch["labels"].numpy())
        preds = np.concatenate(preds, axis=0)

        # the order of the predicitons must be the same as the class_list, which is sorted version of the list
        labels = np.concatenate(labels, axis=0)

        # NOTE: for label only.
        # class_list = sorted(getattr(constants, test_dataset_name.upper()))
        class_list = getattr(constants, test_dataset_name.upper())

        results = {}
        if test_dataset_name in {
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
            results['multilabel_classification'] = {}
            results["multilabel_classification"][test_dataset_name] = multilabel_classification(preds, labels, class_list)
        if test_dataset_name in {"chexpert5x200"}:
            results['multiclass_classification'] = {}
            results["multiclass_classification"][test_dataset_name] = multiclass_classification(preds, labels, class_list)
        return results