import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator



class AdvLiftEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(AdvLiftEvaluator, self).__init__(config)

    def eval_advlift(self,
                     net: nn.Module,
                     id_data_loader: DataLoader,
                     postprocessor: BasePostprocessor,
                    ) -> dict:
        net.eval()
        
        pred_clean_list, pred_adv_list, E_clean_list, E_adv_list, conf_list, label_list = postprocessor.inference(net, id_data_loader, progress=True)
        clean_correct_list = (pred_clean_list == label_list)            # rate = clean_acc
        clean_wrong_list = (pred_clean_list != label_list)              # rate = 1-clean_acc
        flip_success_list = (pred_clean_list != pred_adv_list)          # rate = FR
        flip_fail_list = (pred_clean_list == pred_adv_list)             # rate = 1-FR
        attack_success_list = (clean_correct_list & flip_success_list)  # rate = ASR
        attack_fail_list = (clean_correct_list & flip_fail_list)        # rate = robust_acc
        
        total_samples = len(conf_list)
        classes = np.sort(np.unique(label_list))

        # boolean masks for each group
        group_masks = {
            "overall": np.ones(total_samples, dtype=bool),
            "clean-correct": clean_correct_list,
            "clean-wrong": clean_wrong_list,
            "flip-success": flip_success_list,
            "flip-fail": flip_fail_list,
            "attack-success": attack_success_list,
            "attack-fail": attack_fail_list,
        }

        class_masks = {f"class_{i}": (label_list == i) for i in classes}
        
        group_rate_descriptions = {
            "overall": "100%",
            "clean-correct": "clean_acc",
            "clean-wrong": "1-clean_acc",
            "flip-success": "FR",
            "flip-fail": "1-FR",
            "attack-success": "ASR",
            "attack-fail": "robust_acc",
        }
        
        groupwise_metrics = dict()
        for group_name, group_mask in group_masks.items():
            groupwise_metrics[group_name] = {
                "advlift": conf_list[group_mask],
                "rate": group_mask.mean(),
                "rate_description": group_rate_descriptions[group_name],
            }

        classwise_metrics = dict()
        for class_name, class_mask in class_masks.items():
            classwise_metrics[class_name] = {
                "advlift": conf_list[class_mask],
                "rate": class_mask.mean(),
                "rate_description": class_name,
            }

        # saving raw results
        df = pd.DataFrame({
            "sample_id": np.arange(total_samples),
            "label": label_list,
            "clean_pred": pred_clean_list,
            "adv_pred": pred_adv_list,
            "clean_correct": clean_correct_list,
            "clean_wrong": clean_wrong_list,
            "flip_success": flip_success_list,
            "flip_fail": flip_fail_list,
            "attack_success": attack_success_list,
            "attack_fail": attack_fail_list,
            "E_clean": E_clean_list,
            "E_adv": E_adv_list,
            "advlift": conf_list,
        })
        
        # Add config parameters as constant columns
        common_args = self.config.adversary.common_args
        pgd_args = self.config.adversary.pgd_args
        
        df["adversary"] = self.config.adversary.name
        df["attacked_label"] = common_args.label_source
        df["eps"] = common_args.eps
        if self.config.adversary.name == "pgd":
            df["step_size"] = pgd_args.step_size
            df["steps"] = pgd_args.steps
            df["random_start"] = pgd_args.random_start
        else:
            df["step_size"] = None
            df["steps"] = None
            df["random_start"] = None
        df["seed"] = self.config.seed
        df["model"] = self.config.network.name
        df["checkpoint"] = self.config.network.checkpoint
        df["dataset"] = self.config.dataset.name
        
        # Save to Parquet and CSV
        save_dir = self.config.output_dir
        save_path_parquet = save_dir + "/raw_results.parquet"
        save_path_csv = save_dir + "/raw_results.csv"
        df.to_parquet(save_path_parquet, index=False)
        df.to_csv(save_path_csv, index=False)
        print(f"Per-sample results saved to {save_path_parquet} and {save_path_csv}")

        return groupwise_metrics, classwise_metrics