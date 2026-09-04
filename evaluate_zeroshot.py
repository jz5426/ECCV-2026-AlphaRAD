import glob
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from cxrclip import seed_everything
from cxrclip.evaluator import Evaluator
log = logging.getLogger(__name__)

def print_evals(evals, metric="Accuracy(Micro)", best="max"):   
    keys = sorted(list(evals.values())[0].keys())
    st = "| model | " + " | ".join(keys) + "|\n"
    st += "| :---- | " + " | ".join([("-" * (len(k) - 1)) + ":" for k in keys]) + "|\n"
    if best == "max":
        best_score = 0.0
    elif best == "min":
        best_score = 9e9
    else:
        raise ValueError("Unknown value for best, got %s" % best)
    best_ckpt = None
    for c, e in evals.items():
        filename = ".".join(c.split("/")[-1].split(".")[:-1])
        st += f"| {filename} | " + " | ".join([f"{e[k]:.3f}" for k in keys]) + " |\n"
        cur_score = e[metric]
        if best == "max":
            if best_score <= cur_score:
                best_score = cur_score
                best_ckpt = filename
        elif best == "min":
            if best_score >= cur_score:
                best_score = cur_score
                best_ckpt = filename
    st += f"Best {metric}: {best_score:.3f}, from {best_ckpt}\n"
    return st

@hydra.main(version_base=None, config_path="configs", config_name="eval_zeroshot")
def main(cfg: DictConfig):
    print('checkpoint path: ', cfg.test.checkpoint)
    seed_everything(cfg.test.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OmegaConf.resolve(cfg)
    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))

    cfg_dict = OmegaConf.to_container(cfg)
    evaluator = Evaluator(cfg_dict, ckpt_paths, "test")

    # for each dataset, evaluate each checkpoint.
    per_dataset_eval = []
    for test_dataset_name in evaluator.data_loader_dict.keys():
        # this is the main code for evaluation
        print(test_dataset_name)
        evals = {c: evaluator.evaluate_clip_offline(c, test_dataset_name) for c in ckpt_paths}
        per_dataset_eval.append(evals)

        print("print best score")
        st = ""

        if test_dataset_name in {
            'chestxdet10_gt', 
            'vindr_cxr_gr', 
            'chexlocalize', 
            'node21', 
        }:
            st += f"\nzeroshot pointing game - {test_dataset_name}\n"
            zeroshot_pg = {k: {_k: _v for _k, _v in v["zeroshot_pointing_game"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_pg, metric="Pointing_Game(Avg)", best="max")

        if test_dataset_name in {'ms_cxr_test', 'ms_cxr_test_v2', 'padchest_gr'}:
            st += f"\nzeroshot pointing game - {test_dataset_name}\n"
            zeroshot_pg = {k: {_k: _v for _k, _v in v["zeroshot_refer_grounding"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_pg, metric="zeroshot_refer_grounding(Acc)", best="max")

        if test_dataset_name in {
            'siim_pneumothorax_seg', 
            'chexlocalize_seg',
            'qatacovid_seg'
        }:
            st += f"\nzeroshot segmentation in dice - {test_dataset_name}\n"
            zeroshot_pg = {k: {_k: _v for _k, _v in v["zeroshot_seg"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_pg, metric="zeroshot_segmentation(Avg)", best="max")

        if test_dataset_name in {
            "siim_pneumothorax", 
            "rsna_pneumonia",
            "vindr_cxr", 
            "chest14", 
            "chexpert", 
            "chexpert5x200", 
            "physician_padchest207", 
            "shenzhenxray",
            "cxrlt_task3",
            "cxrlt_task2",
            "montgomery",
            "chestxdet10",
            "chexchonet_composite_slvh_dlv",
            "chexchonet_slvh",  
            "chexchonet_dlv",
            "covidkaggle",
            }:
            st += f"\nzeroshot binary - {test_dataset_name}\n"
            zeroshot_binary = {k: {_k: _v for _k, _v in v["zeroshot_binary"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_binary, metric="AUROC(Avg)", best="max")

        log.info(cfg.test.checkpoint)
        log.info(st)
    return

if __name__ == "__main__":
    main()
