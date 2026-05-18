import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
import openood.utils.comm as comm


class LeakPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.temperature = args.temperature
        self.tau = args.tau
        self.pts_per_bridge = args.pts_per_bridge
        self.lr = args.lr
        self.max_steps = args.max_steps
        
        self.descender = LeakGradDescent(
            lr=self.lr,
            max_steps=self.max_steps,
            temperature=self.temperature
        )

    @torch.no_grad()
    def sample_bridge_points(self, u: torch.Tensor, v: torch.Tensor, n_pts: int):
        """
        Sample n_pts uniformly from [tau, 1-tau] along bridge u--v.
        
        Args:
            u: (1, D) or (D,) tensor
            v: (1, D) or (D,) tensor
            n_pts: number of points to sample
        
        Returns:
            features: (n_pts, D) tensor on same device as u
        """
        u = u.view(1, -1)
        v = v.view(1, -1)
        gammas = torch.rand(n_pts, device=u.device) * (1 - 2*self.tau) + self.tau
        gammas = gammas.unsqueeze(1)  # (n_pts, 1)
        features = (1 - gammas) * u + gammas * v
        return features

    def postprocess(self, net: nn.Module, bridge_pairs):
        """
        Args:
            net: the network
            bridge_pairs: list of (u, v) tensors, each (1, D) on GPU
        
        Returns:
            all_energy_histories: concatenated list of (max_steps+1, pts_per_bridge) numpy arrays into a single (max_steps+1, pts_per_bridge*len(bridge_pairs)) numpy array
        """
        fc = net.get_fc_layer()
        all_energy_histories = list()
        
        for u, v in bridge_pairs:
            features = self.sample_bridge_points(u, v, self.pts_per_bridge)  # (pts, D)
            
            # Run gradient descent
            energy_history = self.descender.descend(fc, features)  # (max_steps+1, pts)
            all_energy_histories.append(energy_history)
        
        return np.concatenate(all_energy_histories, axis=1)

    def inference(self,
                  net: nn.Module,
                  bridge_pairs_per_class_pair,
                  progress: bool = True):
        """
        Args:
            net: the network
            bridge_pairs_per_class_pair: dict {class_pair, bridge_pairs};
                bridge_pairs is a list of (u, v) bridge endpoint tensors
            progress: show progress bar
        
        Returns:
            results_per_class_pair: dict {class_pair, energy_histories}
        """
        net.eval()
        results_per_class_pair = dict()
        
        iterator = tqdm(bridge_pairs_per_class_pair.items(),
                       disable=not progress or not comm.is_main_process(),
                       desc="Leak inference")
        
        for class_pair, bridge_pairs in iterator:
            energy_histories = self.postprocess(net, bridge_pairs)
            results_per_class_pair[class_pair] = energy_histories
        
        return results_per_class_pair

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]

    def get_hyperparam(self):
        return self.temperature

        
class LeakGradDescent:
    def __init__(self, lr, max_steps, temperature):
        self.lr = lr
        self.max_steps = max_steps
        self.temperature = temperature

    def energy_fn(self, features, model_part):
        """E = -T * logsumexp(model_part(features)/T)."""
        logits = model_part(features)
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

    def descend(self, model_part, initial_features):
        """
        Run gradient descent from initial_features.
        
        Args:
            model_part: something to produce logits from the intermediate features. Right now model_part is a final linear layer (callable)
            initial_features: (N, D) tensor on GPU
        
        Returns:
            energy_history: (max_steps+1, N) — energy after 0, 1, ..., max_steps steps
        """
        was_training = model_part.training
        model_part.eval()
        
        features = initial_features.detach().clone()
        energy_history = []

        for step in range(self.max_steps):
            features.requires_grad_(True)
            energy = self.energy_fn(features, model_part)
            energy_history.append(energy.detach().cpu().numpy())

            # sum reduction similar to cross-entropy loss in PGD
            grad = torch.autograd.grad(energy.sum(), features)[0]
            
            features = (features - self.lr * grad).detach()

        # Record energy at the last step
        energy = self.energy_fn(features, model_part)
        energy_history.append(energy.detach().cpu().numpy())

        if was_training:
            model.train()

        # Stack: (max_steps+1, N)
        return np.stack(energy_history, axis=0)