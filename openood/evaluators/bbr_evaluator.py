import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator



class BBREvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(BBREvaluator, self).__init__(config)
        self.tau = self.config.evaluator.evaluator_args.tau
        self.temperature = self.config.evaluator.evaluator_args.temperature
        self.grid_pts = self.config.evaluator.evaluator_args.grid_pts
        self.top_n = config.evaluator.evaluator_args.top_n
        # self.intra_classpair_reduction = config.evaluator.evaluator_args.intra_classpair_reduction
        self.whiten_features = config.evaluator.evaluator_args.whiten_features
        self.whitened_top_n_selection = config.evaluator.evaluator_args.whitened_top_n_selection
        if self.whitened_top_n_selection and not self.whiten_features:
            raise ValueError("Cannot do Top-N selection in whitened features if the features are not whitened in the first place.")

    def total_inference(self, net, dataloader):
        """Return all (features, predictions, labels) as CPU tensors."""
        net.eval()
        feats_list, preds_list, labels_list = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x, y = batch['data'], batch['label']
                else:
                    x, y = batch[0], batch[1]
                x = x.cuda()
                logits, feats = net(x, return_feature=True)
                probas = torch.softmax(logits, dim=1)
                _, preds = torch.max(probas, dim=1)
                feats_list.append(feats.cpu())
                preds_list.append(preds.cpu())
                labels_list.append(y)
        return torch.cat(feats_list, dim=0), torch.cat(preds_list, dim=0), torch.cat(labels_list, dim=0)

    def compute_whitening(self, features, epsilon=1e-6):
        """
        Computes feature mean and ZCA-whitening matrix. Eigvals < eps are treated as zeroes
        
        Args:
            features: shape (n_samples, n_features)
            epsilon: eigvals < eps are treated as zeroes
        """
        mean = features.mean(dim=0)
        centered = features - mean
        
        cov = centered.T @ centered / (features.shape[0] - 1)
        
        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        significant = eigvals > epsilon
        
        sqrt_eigvals = torch.zeros_like(eigvals)
        sqrt_eigvals[significant] = torch.sqrt(eigvals[significant])
        sqrt_eigvals_matrix = torch.diag(sqrt_eigvals)
        restoring_matrix = eigvecs @ sqrt_eigvals_matrix @ eigvecs.T
        
        inv_sqrt_eigvals = torch.zeros_like(eigvals)
        inv_sqrt_eigvals[significant] = 1.0 / torch.sqrt(eigvals[significant])
        inv_sqrt_eigvals_matrix = torch.diag(inv_sqrt_eigvals)
        whitening_matrix = eigvecs @ inv_sqrt_eigvals_matrix @ eigvecs.T

        return whitening_matrix, restoring_matrix, mean

    def whiten(self, features, W, feature_mean):
        return (features - feature_mean) @ W.T

    def restore(self, whitened_features, R, feature_mean):
        "Restores whitened features into their original space"
        return whitened_features @ R.T + feature_mean

    def energy_from_features(self, features, fc, restore: bool, R, mean):
        """return E = -T * logsumexp(fc(features)/T)."""
        if restore:
            features = self.restore(features, R, mean)
        logits = fc(features)
        return -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=1
        )

    def BBR_for_single_class_pair(self, feats_A, feats_B, fc, W, R, feature_mean):
        """
        feats_A : (nA, D) GPU tensor
        feats_B : (nB, D) GPU tensor
        fc      : final linear layer
        returns  scalar BBR metric for this class pair.
        """
        nA, nB = feats_A.shape[0], feats_B.shape[0]

        if self.whiten_features and self.whitened_top_n_selection:
            feats_A, feats_B = self.whiten(feats_A, W, feature_mean), self.whiten(feats_B, W, feature_mean)
            dists = torch.cdist(feats_A, feats_B, p=2)  # (nA, nB)
        elif self.whiten_features and not self.whitened_top_n_selection:
            dists = torch.cdist(feats_A, feats_B, p=2)  # (nA, nB)
            feats_A, feats_B = self.whiten(feats_A, W, feature_mean), self.whiten(feats_B, W, feature_mean)
        else:
            dists = torch.cdist(feats_A, feats_B, p=2)  # (nA, nB)

        # top‑N closest pairs (flatten then pick because torch.topk apparently can't return 2-indices)
        flat_dists = dists.view(-1)
        _, top_indices = torch.topk(flat_dists, k=min(self.top_n, nA * nB), largest=False)
        top_pairs = [(idx // nB, idx % nB) for idx in top_indices.cpu().tolist()]

        # for each pair, find min energy along the line
        pair_values = list()
        curves_values = list()
        tau0_values = list()
        tau1_values = list()
        for iA, iB in top_pairs:
            u = feats_A[iA:iA+1]  # (1, D)
            v = feats_B[iB:iB+1]

            nrg_u = self.energy_from_features(u, fc, self.whiten_features, R, feature_mean).item()
            nrg_v = self.energy_from_features(v, fc, self.whiten_features, R, feature_mean).item()

            tau0_values.append(nrg_u)
            tau1_values.append(nrg_v)

            # dense eval
            coarse_gamma = torch.linspace(self.tau, 1-self.tau, self.grid_pts,
                                          device=u.device).unsqueeze(1)
            interp = (1 - coarse_gamma) * u + coarse_gamma * v
            coarse_nrg = self.energy_from_features(interp, fc, self.whiten_features, R, feature_mean)
            best_idx = torch.argmin(coarse_nrg).item()
            gamma_approx = coarse_gamma[best_idx].item()
            curves_values.append(coarse_nrg.detach().cpu().numpy())
            
            # reeval around the supposed min
            delta = 0.05
            left = max(self.tau, gamma_approx - delta)
            right = min(1.0 - self.tau, gamma_approx + delta)
            fine_gamma = torch.linspace(left, right, self.grid_pts,
                                        device=u.device).unsqueeze(1)
            interp = (1 - fine_gamma) * u + fine_gamma * v
            nrg = self.energy_from_features(interp, fc, self.whiten_features, R, feature_mean)
            min_nrg = torch.min(nrg).item()

            bbr_val = min_nrg - 0.5 * (nrg_u + nrg_v)
            pair_values.append(bbr_val)

        # intra‑classpair reductions and values for plotting
        return {
            "min": float(np.min(pair_values)),
            "mean": float(np.mean(pair_values)),
            "raw": pair_values,
            "curves": curves_values,
            "tau0s": tau0_values,
            "tau1s": tau1_values
        }

    def eval_bbr(self, net, id_data_loader, W, R, feature_mean):
        features, preds, labels = self.total_inference(net, id_data_loader)
        labels = labels.numpy()
        classes = np.unique(labels)
        fc = net.get_fc_layer()

        # only features that led to correct predictions will be used
        correct_mask = (preds.numpy() == labels)

        results_dict = {}

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                cls_i_mask = (labels == classes[i]) & correct_mask
                cls_j_mask = (labels == classes[j]) & correct_mask

                feats_i = features[cls_i_mask].cuda()
                feats_j = features[cls_j_mask].cuda()

                if feats_i.shape[0] == 0 or feats_j.shape[0] == 0:
                    continue

                single_pair_dict = self.BBR_for_single_class_pair(
                    feats_i, feats_j, fc,
                    # reduction=self.intra_classpair_reduction,
                    W.cuda(), R.cuda(), feature_mean.cuda()
                )
                results_dict[(classes[i], classes[j])] = single_pair_dict

        print(f"BBR interclass means over {len(results_dict.keys())} class pairs:")
        print(f"Intraclass reduction = MEAN: {sum([results_dict[class_pair]["mean"] for class_pair in results_dict.keys()]) / len(results_dict.keys())}")
        print(f"Intraclass reduction = MIN: {sum([results_dict[class_pair]["min"] for class_pair in results_dict.keys()]) / len(results_dict.keys())}")

        self.plot_results(results_dict)
        self.results_dict_to_parquet(results_dict)
        self.aggregated_results_dict_to_csv(results_dict)

        return results_dict

    def plot_results(self, results_dict):
        x_start = self.tau
        x_end = 1 - self.tau
        n_points = self.grid_pts
        x_values = np.linspace(x_start, x_end, n_points)
        
        # 1. Create individual plots for each key (n_keys plots)
        for ith_key, data in results_dict.items():
            plt.figure(figsize=(10, 6))
            
            # Plot all curves for this key
            for i in range(len(data["curves"])):
                # Insert tau0 at beginning and tau1 at end
                color = np.random.rand(3)
                plt.plot(0, data["tau0s"][i], marker='x', markersize=10, alpha=0.7, color=color)
                plt.plot(1, data["tau1s"][i], marker='x', markersize=10, alpha=0.7, color=color)
                plt.plot(x_values, data["curves"][i], alpha=0.7, color=color)
            
            plt.xlabel('tau')
            plt.ylabel('Energy level')
            plt.title(f'Curves for {ith_key}')
            plt.grid(True, alpha=0.3)
            
            # Set x-axis limits
            plt.xlim(-0.025, 1.025)
            
            # Save the plot
            # Convert tuple key to string for filename
            if isinstance(ith_key, tuple):
                key_str = '_'.join(str(k) for k in ith_key)
            else:
                key_str = str(ith_key)
            plt.savefig(f"{self.config.output_dir}/{key_str}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Create aggregated mean plot
        plt.figure(figsize=(10, 6))
        all_mean_curves = list()
        all_mean_tau0s, all_mean_tau1s = list(), list()
        
        for ith_key, data in results_dict.items():
            # Compute element-wise mean across all curves for this key
            curves_array = np.array(data["curves"])
            mean_curve = np.mean(curves_array, axis=0)

            tau0s_array = np.array(data["tau0s"])
            mean_tau0 = np.mean(tau0s_array, axis=0)
            tau1s_array = np.array(data["tau1s"])
            mean_tau1 = np.mean(tau1s_array, axis=0)
            
            all_mean_curves.append(mean_curve)
            all_mean_tau0s.append(mean_tau0)
            all_mean_tau1s.append(mean_tau1)
            
            # Plot
            if isinstance(ith_key, tuple):
                label = '_'.join(str(k) for k in ith_key)
            else:
                label = str(ith_key)
            color = np.random.rand(3)
            plt.plot(0, mean_tau0, marker='x', markersize=10, alpha=0.7, color=color)#, label=label)
            plt.plot(1, mean_tau1, marker='x', markersize=10, alpha=0.7, color=color)#, label=label)
            plt.plot(x_values, mean_curve, alpha=0.7, color=color)#, label=label)
        
        plt.xlabel('tau')
        plt.ylabel('Energy level (mean)')
        plt.title('Aggregated Mean Curves for All Class Pairs')
        # plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.025, 1.025)
        plt.savefig(f"{self.config.output_dir}/all_class_pairs_aggregated_mean.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Create aggregated min plot
        plt.figure(figsize=(10, 6))
        
        for ith_key, data in results_dict.items():
            # Compute element-wise min across all curves for this key
            curves_array = np.array(data["curves"])
            min_curve = np.min(curves_array, axis=0)

            tau0s_array = np.array(data["tau0s"])
            min_tau0 = np.min(tau0s_array, axis=0)
            tau1s_array = np.array(data["tau1s"])
            min_tau1 = np.min(tau1s_array, axis=0)
            
            # Plot
            if isinstance(ith_key, tuple):
                label = '_'.join(str(k) for k in ith_key)
            else:
                label = str(ith_key)
            color = np.random.rand(3)
            plt.plot(0, min_tau0, marker='x', markersize=10, alpha=0.7, color=color)#, label=label)
            plt.plot(1, min_tau1, marker='x', markersize=10, alpha=0.7, color=color)#, label=label)
            plt.plot(x_values, min_curve, alpha=0.7, color=color)#, label=label)
        
        plt.xlabel('tau')
        plt.ylabel('Energy level (min)')
        plt.title('Aggregated Min Curves for All Class Pairs')
        # plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.025, 1.025)
        plt.savefig(f"{self.config.output_dir}/all_class_pairs_aggregated_min.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created {len(results_dict)} individual plots and 2 aggregated plots")

    def results_dict_to_parquet(self, results_dict):
        rows = []
        for (class0, class1), data in results_dict.items():
            for i in range(len(data["curves"])):
                rows.append({
                    "class0": class0,
                    "class1": class1,
                    "min_on_curve": data["curves"][i].min(),
                    "raw_bbr": data["raw"][i],
                    "curve": data["curves"][i],
                    "tau0": data["tau0s"][i],
                    "tau1": data["tau1s"][i]
                })
        
        df = pd.DataFrame(rows)
        df["whitened"] = self.whiten_features
        df["tau"] = self.tau
        df["grid_pts"] = self.grid_pts
        # df["curves_n"] = self.top_n
        df["seed"] = self.config.seed
        df["model"] = self.config.network.name
        df["checkpoint"] = self.config.network.checkpoint
        df["dataset"] = self.config.dataset.name
        
        df.to_parquet(f"{self.config.output_dir}/raw_results.parquet", index=False)

    def aggregated_results_dict_to_csv(self, results_dict):
        rows = []
        for (class0, class1), data in results_dict.items():
            rows.append({
                    "class0": class0,
                    "class1": class1,
                    "min_bbr": data["min"],
                    "mean_bbr": data["mean"]
            })
        
        df = pd.DataFrame(rows)
        df["whitened"] = self.whiten_features
        df["tau"] = self.tau
        df["grid_pts"] = self.grid_pts
        df["curves_n"] = self.top_n
        df["seed"] = self.config.seed
        df["model"] = self.config.network.name
        df["checkpoint"] = self.config.network.checkpoint
        df["dataset"] = self.config.dataset.name
        
        df.to_csv(f"{self.config.output_dir}/aggregated_results.csv", index=False)

        rows = [{
            "mean_reduction": sum([results_dict[class_pair]["mean"] for class_pair in results_dict.keys()]) / len(results_dict.keys()),
            "min_reduction": sum([results_dict[class_pair]["min"] for class_pair in results_dict.keys()]) / len(results_dict.keys())
        }]

        df = pd.DataFrame(rows)
        df["whitened"] = self.whiten_features
        df["tau"] = self.tau
        df["grid_pts"] = self.grid_pts
        df["curves_n"] = self.top_n
        df["seed"] = self.config.seed
        df["model"] = self.config.network.name
        df["checkpoint"] = self.config.network.checkpoint
        df["dataset"] = self.config.dataset.name

        df.to_csv(f"{self.config.output_dir}/global_results.csv", index=False)