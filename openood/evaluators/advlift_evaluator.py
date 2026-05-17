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
        """
        Compute AdvLift = [E(f(x_adv)) - E(f(x))] distribution, attack success rate (ASR) and flip rate (FR).

        FR is computed as a fraction of all the predictions that were successfully flipped, while ASR considers previously correct predictions only.
        """
        net.eval()
        
        pred_clean_list, pred_adv_list, E_clean_list, E_adv_list, conf_list, label_list = postprocessor.inference(net, id_data_loader, progress=True)
        flip_list = (pred_clean_list != pred_adv_list)
        clean_correct_list = (pred_clean_list == label_list)
        adv_correct_list = (pred_adv_list == label_list)
        attack_success_list = (clean_correct_list & flip_list)
        
        fr = float(np.mean(flip_list))
        asr = float(np.mean(flip_list[clean_correct_list]))
        clean_acc = float(np.mean(clean_correct_list))
        robust_acc = float(np.mean(clean_correct_list & adv_correct_list))  # i.e. if an example was failed while clean, it won't count after attack
        
        metrics = {"advlift": conf_list, "asr": asr, "fr": fr, "clean_acc": clean_acc, "robust_acc": robust_acc}

        n_samples = len(conf_list)
        df = pd.DataFrame({
            "sample_id": np.arange(n_samples),
            "label": label_list,
            "clean_pred": pred_clean_list,
            "adv_pred": pred_adv_list,
            "clean_correct": clean_correct_list,
            "attack_flipped": flip_list,
            "attack_successful": attack_success_list,
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
        if self.config.adversary.name == "fgsm":
            df["step_size"] = None
            df["steps"] = None
            df["random_start"] = None
        else:
            df["step_size"] = pgd_args.step_size
            df["steps"] = pgd_args.steps
            df["random_start"] = pgd_args.random_start
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

        # print(f"AdvLift (mean): {float(np.mean(conf_list))}, ASR: {(100*asr):.2f}%, FR: {(100*fr):.2f}%, ACC (clean): {clean_acc:.5f}, ACC (robust): {robust_acc:.5f}")
        return metrics
