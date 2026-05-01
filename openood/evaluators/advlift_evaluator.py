import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator



class AdvLiftEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """AdvLift Evaluator.

        Args:
            config (Config): Config file
        """
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
        
        pred_clean_list, pred_adv_list, conf_list, label_list = postprocessor.inference(net, id_data_loader, progress=True)
        flip_list = (pred_clean_list != pred_adv_list)
        clean_correct_list = (pred_clean_list == label_list)
        adv_correct_list = (pred_adv_list == label_list)
        
        fr = float(np.mean(flip_list))
        asr = float(np.mean(flip_list[clean_correct_list]))
        clean_acc = float(np.mean(clean_correct_list))
        robust_acc = float(np.mean(clean_correct_list & adv_correct_list))  # i.e. if an example was failed while clean, it won't count after attack
        
        metrics = {"advlift": conf_list, "asr": asr, "fr": fr, "clean_acc": clean_acc, "robust_acc": robust_acc}

        print(f"AdvLift (mean): {float(np.mean(conf_list))}, ASR: {(100*asr):.2f}%, FR: {(100*fr):.2f}%, ACC (clean): {clean_acc:.5f}, ACC (robust): {robust_acc:.5f}")
        return metrics