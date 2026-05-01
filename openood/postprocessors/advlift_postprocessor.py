from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_postprocessor import BasePostprocessor

import openood.utils.comm as comm

from openood.preprocessors.transform import normalization_dict


class AdvLiftPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.adv_args = self.config.adversary.adversary_args
        self.temperature = self.args.temperature

        self.input_mean, self.input_std = normalization_dict[self.config.dataset.name]
        # [0, 1] -> [PIXEL_MIN, PIXEL_MAX] in each color channel separately
        self.pixel_min = torch.tensor([(0 - m) / s for m, s in zip(self.input_mean, self.input_std)])
        self.pixel_max = torch.tensor([(1 - m) / s for m, s in zip(self.input_mean, self.input_std)])

        adversaries = {
            "fgsm": FGSM(self.adv_args.alpha/255, F.cross_entropy, self.pixel_min, self.pixel_max),
            "pgd": PGD(self.adv_args.alpha/255, self.adv_args.epsilon/255, self.adv_args.steps, F.cross_entropy, self.pixel_min, self.pixel_max)
        }

        self.adversary = adversaries[self.config.adversary.name]
        
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any):
        with torch.no_grad():
            logits_clean = net(data)
            probas_clean = torch.softmax(logits_clean, dim=1)
            _, pred_clean = torch.max(probas_clean, dim=1)

        data_adv = self.adversary.attack(net, data, pred_clean)
        
        with torch.no_grad():
            logits_adv = net(data_adv)
            probas_adv = torch.softmax(logits_adv, dim=1)
            _, pred_adv = torch.max(probas_adv, dim=1)
        
        E_clean = -self.temperature * torch.logsumexp(logits_clean / self.temperature, dim=1)
        E_adv = -self.temperature * torch.logsumexp(logits_adv / self.temperature, dim=1)
        conf = E_adv - E_clean
        return pred_clean, pred_adv, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_clean_list, pred_adv_list, conf_list, label_list = [], [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred_clean, pred_adv, conf = self.postprocess(net, data)

            pred_clean_list.append(pred_clean.cpu())
            pred_adv_list.append(pred_adv.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_clean_list = torch.cat(pred_clean_list).numpy().astype(int)
        pred_adv_list = torch.cat(pred_adv_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_clean_list, pred_adv_list, conf_list, label_list


class FGSM:
    def __init__(self, alpha, cls_loss_fn, pixel_min, pixel_max):
        self.alpha = alpha
        self.cls_loss_fn = cls_loss_fn
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def attack(self, model, x, y):
        # note: it is an untargeted one!
        was_training = model.training
        model.eval()
    
        x_orig = x.detach().clone().requires_grad_(True)
        logits = model(x_orig)
        loss = self.cls_loss_fn(logits, y)
        # model.zero_grad() -- not needed anymore if torch.autograd.grad() is operating
        # loss.backward() -- replaced by torch.autograd.grad()
        dL_dx = torch.autograd.grad(loss, x_orig)[0]
    
        min_vals = self.pixel_min.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)
        max_vals = self.pixel_max.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)
    
        ranges = max_vals - min_vals
    
        x_adv = x_orig + self.alpha * ranges * dL_dx.sign()
        x_adv = torch.clamp(x_adv, min_vals, max_vals)
    
        if was_training:
            model.train()
    
        return x_adv.detach()



class PGD:
    def __init__(self, alpha, epsilon, steps, cls_loss_fn, pixel_min, pixel_max):
        self.alpha = alpha      # ultimate boundary
        self.epsilon = epsilon  # single step boundary
        self.steps = steps
        self.cls_loss_fn = cls_loss_fn
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def attack(self, model, x, y):
        was_training = model.training
        model.eval()

        min_vals = self.pixel_min.to(x.device).view(1, 3, 1, 1).expand_as(x)
        max_vals = self.pixel_max.to(x.device).view(1, 3, 1, 1).expand_as(x)
    
        ranges = max_vals - min_vals

        min_vals_alpha = torch.clamp(x - self.alpha * ranges, min_vals, max_vals)
        max_vals_alpha = torch.clamp(x + self.alpha * ranges, min_vals, max_vals)
    
        x_adv = x.detach().clone()
        uniform01 = torch.rand(x_adv.size(), device=x_adv.device)
        x_adv += uniform01 * 2 * ranges * self.alpha - ranges * self.alpha
        x_adv.requires_grad_(True)

        for i in range(self.steps):
            logits = model(x_adv)
            loss = self.cls_loss_fn(logits, y)
            dL_dx = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + self.epsilon * ranges * dL_dx.sign()
            x_adv = torch.clamp(x_adv, min_vals_alpha, max_vals_alpha)

        if was_training:
            model.train()
    
        return x_adv.detach()
            