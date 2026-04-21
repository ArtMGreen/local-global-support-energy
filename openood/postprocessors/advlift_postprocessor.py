from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor

from openood.preprocessors.transform import normalization_dict


class AdvLiftPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature

        self.input_mean, self.input_std = normalization_dict[self.config.dataset.name]
        # [0, 1] -> [PIXEL_MIN, PIXEL_MAX] in each color channel separately
        self.pixel_min = torch.tensor([(0 - m) / s for m, s in zip(self.input_mean, self.input_std)])
        self.pixel_max = torch.tensor([(1 - m) / s for m, s in zip(self.input_mean, self.input_std)])

        self.adversary = FGSM(self.args.alpha/255, F.cross_entropy, self.pixel_min, self.pixel_max)
        
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
        return (pred_clean != pred_adv), conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]

    def get_hyperparam(self):
        return self.temperature


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