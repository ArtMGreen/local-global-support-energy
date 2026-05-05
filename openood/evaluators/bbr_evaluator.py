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
        self.top_n = config.evaluator.evaluator_args.top_n
        self.intra_classpair_reduction = config.evaluator.evaluator_args.intra_classpair_reduction

    def total_inference(self, net, dataloader):
        """Return all (features, predictions, labels) as CPU tensors."""
        net.eval()
        feats_list, preds_list, labels_list = [], []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x, y = batch['data'], batch['label']
                else:
                    x, y = batch[0], batch[1]
                x = x.cuda()
                logits, feats = net(x, return_feature=True)  # (logits, features)
                probas = torch.softmax(logits, dim=1)
                _, preds = torch.max(probas, dim=1)
                feats_list.append(feats.cpu())
                preds_list.append(preds.cpu())
                labels_list.append(y)
        return torch.cat(feats_list, dim=0), torch.cat(preds_list, dim=0), torch.cat(labels_list, dim=0)

    def energy_from_features(self, features, fc):
        """return E = -T * logsumexp(fc(features)/T)."""
        logits = fc(features)
        return -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=1
        )

    def BBR_for_single_class_pair(self, feats_A, feats_B, fc, reduction="mean"):
        """
        feats_A : (nA, D) GPU tensor
        feats_B : (nB, D) GPU tensor
        fc      : final linear layer
        returns  scalar BBR metric for this class pair.
        """
        nA, nB = feats_A.shape[0], feats_B.shape[0]

        # pairwise Euclidean distances (on GPU for speed)
        dists = torch.cdist(feats_A, feats_B, p=2)  # (nA, nB)

        # top‑N closest pairs (flatten then pick because torch.topk apparently can't return 2-indices)
        flat_dists = dists.view(-1)
        _, top_indices = torch.topk(flat_dists, k=min(self.top_n, nA * nB), largest=False)
        top_pairs = [(idx // nB, idx % nB) for idx in top_indices.cpu().tolist()]

        # for each pair, find min energy along the line
        pair_values = list()
        for iA, iB in top_pairs:
            u = feats_A[iA:iA+1]  # (1, D)
            v = feats_B[iB:iB+1]

            nrg_u = self.energy_from_features(u, fc).item()
            nrg_v = self.energy_from_features(v, fc).item()

            # dense eval
            coarse_gamma = torch.linspace(self.tau, 1-self.tau, self.grid_pts,
                                          device=u.device).unsqueeze(1)
            interp = (1 - coarse_gamma) * u + coarse_gamma * v
            nrg = self.energy_from_features(interp, fc)
            best_idx = torch.argmin(nrg).item()
            gamma_approx = coarse_gamma[best_idx].item()
            
            # reeval around the supposed min
            delta = 0.05
            left = max(self.tau, gamma_approx - delta)
            right = min(1.0 - self.tau, gamma_approx + delta)
            fine_gamma = torch.linspace(left, right, self.grid_pts,
                                        device=u.device).unsqueeze(1)
            interp = (1 - fine_gamma) * u + fine_gamma * v
            nrg = self.energy_from_features(interp, fc)
            min_nrg = torch.min(nrg).item()

            bbr_val = min_nrg - 0.5 * (nrg_u + nrg_v)
            pair_values.append(bbr_val)

        # intra‑classpair reduction
        if reduction == "min":
            return float(np.min(pair_values))
        elif reduction == "mean":
            return float(np.mean(pair_values))
        else:
            raise ValueError('reduction can only be "mean" or "min"')

    def eval_bbr(self, net, id_data_loader):
        features, preds, labels = self.total_inference(net, id_data_loader)
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

            # reevaluation around best

            bbr_val = min_nrg - 0.5 * (nrg_u + nrg_v)
            values.append(bbr_val)
            values_dict[class_nums] = bbr_val

        bbr_mean = float(np.mean(values))
        print(f"BBR over {len(values)} cross‑class pairs: {bbr_mean:.6f}")
        print(*values_dict.items(), sep="\n")
        return {'bbr': bbr_mean, 'bbr_by_pairs': values_dict, 'curves_by_pairs': curves_dict}

    def eval_bbr(self, net, id_data_loader):
        features, preds, labels = self.total_inference(net, id_data_loader)
        labels = labels.numpy()
        classes = np.unique(labels)
        fc = net.get_fc_layer()

        # only features that led to correct predictions will be used
        correct_mask = (preds.numpy() == labels)

        values = []
        values_dict = {}

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                cls_i_mask = (labels == classes[i]) & correct_mask
                cls_j_mask = (labels == classes[j]) & correct_mask

                feats_i = features[cls_i_mask].cuda()
                feats_j = features[cls_j_mask].cuda()

                if feats_i.shape[0] == 0 or feats_j.shape[0] == 0:
                    continue

                bbr_val = self.BBR_for_single_class_pair(
                    feats_i, feats_j, fc,
                    reduction=self.intra_classpair_reduction
                )
                values.append(bbr_val)
                values_dict[(classes[i], classes[j])] = bbr_val

        # 3. Inter‑class‑pair reduction (always mean)
        bbr_mean = float(np.mean(values)) if values else 0.0
        print(f"BBR ({INTRA_CLASSPAIR_REDUCTION}), mean over {len(values)} class pairs: {bbr_mean:.6f}")
        for k, v in values_dict.items():
            print(f"  {k}: {v:.6f}")

        return {
            'bbr': bbr_mean,
            'bbr_by_pairs': values_dict
        }