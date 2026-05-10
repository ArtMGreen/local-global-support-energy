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
        self.common_args = self.config.adversary.common_args
        self.pgd_args = self.config.adversary.pgd_args
        self.temperature = self.args.temperature

        self.input_mean, self.input_std = normalization_dict[self.config.dataset.name]
        # [0, 1] -> [PIXEL_MIN, PIXEL_MAX] in each color channel separately
        self.pixel_min = torch.tensor([(0 - m) / s for m, s in zip(self.input_mean, self.input_std)])
        self.pixel_max = torch.tensor([(1 - m) / s for m, s in zip(self.input_mean, self.input_std)])

        adversaries = {
            "fgsm": FGSM(self.common_args.eps/255, F.cross_entropy, self.pixel_min, self.pixel_max),
            "pgd": PGD(self.common_args.eps/255, self.pgd_args.step_size/255, self.pgd_args.steps, self.pgd_args.random_start, F.cross_entropy, self.pixel_min, self.pixel_max)
        }

        self.adversary = adversaries[self.config.adversary.name]
        print("Adversary info:")
        print(self.adversary.description())
        print("Label source:", self.common_args.label_source)
        
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any, label):
        with torch.no_grad():
            logits_clean = net(data)
            probas_clean = torch.softmax(logits_clean, dim=1)
            _, pred_clean = torch.max(probas_clean, dim=1)

        if label is None:
            data_adv = self.adversary.attack(net, data, pred_clean)
        else:
            data_adv = self.adversary.attack(net, data, label)
        
        with torch.no_grad():
            logits_adv = net(data_adv)
            probas_adv = torch.softmax(logits_adv, dim=1)
            _, pred_adv = torch.max(probas_adv, dim=1)
        
        E_clean = -self.temperature * torch.logsumexp(logits_clean / self.temperature, dim=1)
        E_adv = -self.temperature * torch.logsumexp(logits_adv / self.temperature, dim=1)
        conf = E_adv - E_clean
        return pred_clean, pred_adv, E_clean, E_adv, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_clean_list, pred_adv_list, E_clean_list, E_adv_list, conf_list, label_list = [], [], [], [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()

            if self.common_args.label_source == "prediction":
                pred_clean, pred_adv, E_clean, E_adv, conf = self.postprocess(net, data, None)
            elif self.common_args.label_source == "ground":
                pred_clean, pred_adv, E_clean, E_adv, conf = self.postprocess(net, data, label)
            else:
                raise ValueError("label_source can only be 'ground' or 'prediction'")

            pred_clean_list.append(pred_clean.cpu())
            pred_adv_list.append(pred_adv.cpu())
            E_clean_list.append(E_clean.cpu())
            E_adv_list.append(E_adv.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_clean_list = torch.cat(pred_clean_list).numpy().astype(int)
        pred_adv_list = torch.cat(pred_adv_list).numpy().astype(int)
        E_clean_list = torch.cat(E_clean_list).numpy()
        E_adv_list = torch.cat(E_adv_list).numpy()
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_clean_list, pred_adv_list, E_clean_list, E_adv_list, conf_list, label_list


class FGSM:
    def __init__(self, eps, cls_loss_fn, pixel_min, pixel_max):
        self.eps = eps
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
    
        x_adv = x_orig + self.eps * ranges * dL_dx.sign()
        x_adv = torch.clamp(x_adv, min_vals, max_vals)
    
        if was_training:
            model.train()
    
        return x_adv.detach()

    def description(self):
        return f"FGSM(ε={(self.eps):.5f}≈{(self.eps * 255):.1f}/255)"



class PGD:
    def __init__(self, eps, step_size, steps, random_start, cls_loss_fn, pixel_min, pixel_max):
        self.eps = eps    # ultimate boundary
        self.step_size = step_size
        self.steps = steps
        self.random_start = random_start
        self.cls_loss_fn = cls_loss_fn
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def attack(self, model, x, y):
        was_training = model.training
        model.eval()

        min_vals = self.pixel_min.to(x.device).view(1, 3, 1, 1).expand_as(x)
        max_vals = self.pixel_max.to(x.device).view(1, 3, 1, 1).expand_as(x)
    
        ranges = max_vals - min_vals

        min_vals_eps = torch.clamp(x - self.eps * ranges, min_vals, max_vals)
        max_vals_eps = torch.clamp(x + self.eps * ranges, min_vals, max_vals)
    
        x_adv = x.detach().clone()

        if self.random_start:
            uniform01 = torch.rand(x_adv.size(), device=x_adv.device)
            random_start = (2 * uniform01 - 1) * ranges * self.eps
            x_adv += random_start
            # usually is not supposed to do anything, given the construction of the random start
            x_adv = torch.clamp(x_adv, min_vals_eps, max_vals_eps)

        for i in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = self.cls_loss_fn(logits, y)
            dL_dx = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + self.step_size * ranges * dL_dx.sign()
            x_adv = torch.clamp(x_adv, min_vals_eps, max_vals_eps).detach()

        if was_training:
            model.train()
    
        return x_adv.detach()

    def description(self):
        return f"PGD-{self.steps}(ε={(self.eps):.5f}≈{(self.eps * 255):.1f}/255), step_size={(self.step_size):.5f}≈{(self.step_size * 255):.1f}/255, random_start={self.random_start}"
