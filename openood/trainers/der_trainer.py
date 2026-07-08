import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config
from openood.preprocessors.transform import normalization_dict

from .lr_scheduler import cosine_annealing


class DERTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.args = self.config.trainer.trainer_args
        self.gamma = self.args.gamma
        self.beta = self.args.beta

        self.input_mean, self.input_std = normalization_dict[self.config.dataset.name]
        # [0, 1] -> [PIXEL_MIN, PIXEL_MAX] in each color channel separately
        self.pixel_min = torch.tensor([(0 - m) / s for m, s in zip(self.input_mean, self.input_std)])
        self.pixel_max = torch.tensor([(1 - m) / s for m, s in zip(self.input_mean, self.input_std)])

        # self.adversary = FGSM(8/255, F.cross_entropy, self.pixel_min, self.pixel_max)
        self.adversary = PGD(8/255, 1/255, 12, True, F.cross_entropy, self.pixel_min, self.pixel_max)

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lambda step: cosine_annealing(
        #         step,
        #         config.optimizer.num_epochs * len(train_loader),
        #         1,
        #         1e-6 / config.optimizer.lr,
        #     ),
        # )

    def train_epoch(self, epoch_idx):
        #with torch.autograd.detect_anomaly():
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_clean = self.net(data)
            loss_ce_clean = F.cross_entropy(logits_clean, target)

            data_adv = self.adversary.attack(self.net, data, target)
            logits_adv = self.net(data_adv)
            loss_ce_adv = F.cross_entropy(logits_adv, target)

            batch_indices = torch.arange(target.size(0), device=target.device)
            
            correct_logits_clean = logits_clean[batch_indices, target]
            correct_logits_adv = logits_adv[batch_indices, target]
            # Delta of joint energy
            E_joint = -correct_logits_clean+correct_logits_adv
            # Delta of Helmholtz (free) energy
            E_marginal = -torch.logsumexp(logits_clean, dim=1)+torch.logsumexp(logits_adv, dim=1)
            
            der_penalty = torch.sqrt(E_joint**2 + E_marginal**2 + 1e-8)
            loss_der = torch.clamp(der_penalty - self.gamma, min=0).mean()
            # print(E_joint.mean().item(), E_marginal.mean().item(), der_penalty.mean().item(), loss_der.item(), sep="\t")
            
            loss = loss_ce_clean+loss_ce_adv+self.beta*loss_der

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                # print(loss_avg)
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


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