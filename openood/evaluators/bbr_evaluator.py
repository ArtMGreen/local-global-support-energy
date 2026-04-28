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

from sklearn.neighbors import NearestNeighbors

from .base_evaluator import BaseEvaluator



class BBREvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """AdvLift Evaluator.

        Args:
            config (Config): Config file
        """
        super(BBREvaluator, self).__init__(config)
        self.tau = self.config.evaluator.evaluator_args.tau
        self.temperature = self.config.evaluator.evaluator_args.temperature
        self.grid_pts = self.config.evaluator.evaluator_args.grid_pts

    def extract_features(self, net, dataloader):
        """Return all (features, labels) as CPU tensors."""
        net.eval()
        feats_list, labels_list = [], []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x, y = batch['data'], batch['label']
                else:
                    x, y = batch[0], batch[1]
                x = x.cuda()
                _, f = net(x, return_feature=True)  # (logits, features)
                feats_list.append(f.cpu())
                labels_list.append(y)
        return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)

    def energy_from_features(self, features, fc):
        """return E = -T * logsumexp(fc(features)/T)."""
        logits = fc(features)
        return -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=1
        )

    def iterative_mutual_nearest(self, class_a_feats, class_b_feats):
        """
        Find a suboptimal mutually nearest pair by the iterative scheme:
        1. choose random u in A
        2. v = nearest to u in B
        3. u_new = nearest to v in A
        4. repeat until u_new == u
        Assumes well-behaved clusters ! ! !
        Returns final (u_idx, v_idx, distance).
        """
        # to numpy for sklearn compat
        A = class_a_feats.numpy()
        B = class_b_feats.numpy()
        nA, nB = len(A), len(B)

        nbrs_A = NearestNeighbors(n_neighbors=1).fit(A)
        nbrs_B = NearestNeighbors(n_neighbors=1).fit(B)

        # 1. random starting point in A
        u_idx = np.random.randint(0, nA)
        prev_u_idx = None
        while prev_u_idx != u_idx:
            prev_u_idx = u_idx
            # 2. nearest v
            u_vec = A[u_idx].reshape(1, -1)
            dist, v_idx = nbrs_B.kneighbors(u_vec, return_distance=True)
            v_idx = v_idx[0][0]
            dist = dist[0][0]
            # 3. nearest u from v
            v_vec = B[v_idx].reshape(1, -1)
            _, u_idx_new = nbrs_A.kneighbors(v_vec, return_distance=True)
            u_idx = u_idx_new[0][0]

        return u_idx, v_idx, dist

    def eval_bbr(self, net, id_data_loader):
        features, labels = self.extract_features(net, id_data_loader)
        labels = labels.numpy()
        classes = np.unique(labels)
        fc = net.get_fc_layer()

        bridge_pairs = []   # (u_feat, v_feat) both 1xD tensors on GPU
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                mask_i = labels == classes[i]
                mask_j = labels == classes[j]
                feats_i = features[mask_i]
                feats_j = features[mask_j]

                if len(feats_i) == 0 or len(feats_j) == 0:
                    continue

                u_idx, v_idx, _ = self.iterative_mutual_nearest(feats_i, feats_j)

                u = features[mask_i][u_idx].cuda().unsqueeze(0)  # (1, D)
                v = features[mask_j][v_idx].cuda().unsqueeze(0)
                bridge_pairs.append(((i, j), (u, v)))

        if not bridge_pairs:
            raise RuntimeError("No bridge pairs found.")

        # minimizing energy along the line segment u-to-v
        values = list()
        values_dict = dict()
        curves_dict = dict()
        for class_nums, vectors in bridge_pairs:
            u, v = vectors
            nrg_u = self.energy_from_features(u, fc).item()
            nrg_v = self.energy_from_features(v, fc).item()

            # dense evaluation, grid of N pts
            coarse_gamma = torch.linspace(self.tau, 1-self.tau, self.grid_pts, device=u.device).unsqueeze(1)
            interp = (1 - coarse_gamma) * u + coarse_gamma * v
            nrg = self.energy_from_features(interp, fc)
            print(class_nums, nrg)
            curves_dict[class_nums] = (coarse_gamma.detach().cpu().numpy(), nrg.detach().cpu().numpy())
            best_idx = torch.argmin(nrg).item()
            gamma_approx = coarse_gamma[best_idx].item()

            # reevaluation around best
            delta = 0.05
            left = max(self.tau, gamma_approx - delta)
            right = min(1.0 - self.tau, gamma_approx + delta)
            fine_gamma = torch.linspace(left, right, self.grid_pts, device=u.device).unsqueeze(1)
            interp = (1 - fine_gamma) * u + fine_gamma * v
            nrg = self.energy_from_features(interp, fc)
            min_nrg = torch.min(nrg).item()

            bbr_val = min_nrg - 0.5 * (nrg_u + nrg_v)
            values.append(bbr_val)
            values_dict[class_nums] = bbr_val

        bbr_mean = float(np.mean(values))
        print(f"BBR over {len(values)} cross‑class pairs: {bbr_mean:.6f}")
        print(*values_dict.items(), sep="\n")
        return {'bbr': bbr_mean, 'bbr_by_pairs': values_dict, 'curves_by_pairs': curves_dict}