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
        Compute AdvLift = [E(f(x_adv)) - E(f(x))].mean() over the dataset.
        """
        net.eval()
        
        pred_list, conf_list, _ = postprocessor.inference(net, id_data_loader, progress=True)

        advlift = float(np.mean(conf_list))
        asr = float(np.mean(pred_list))  # ASR - attack success rate
        metrics = {"advlift": advlift, "asr": asr}

        print(f"AdvLift: {advlift}, succ atk rate: {(100*asr):.2f}%")
        return metrics