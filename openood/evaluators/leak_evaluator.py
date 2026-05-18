import numpy as np
import torch

from openood.evaluators.base_evaluator import BaseEvaluator
from openood.utils import Config


class LeakEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super().__init__(config)
        postproc_args = self.config.postprocessor.postprocessor_args
        self.temperature = postproc_args.temperature
        self.top_n = config.evaluator.evaluator_args.top_n
        # self.whiten_features = config.evaluator.evaluator_args.whiten_features
        self.whitened_top_n_selection = config.evaluator.evaluator_args.whitened_top_n_selection

    @torch.no_grad()
    def total_inference(self, net, dataloader):
        """Return all (features, logits, predictions, labels) as CPU tensors."""
        net.eval()
        feats_list, logits_list, preds_list, labels_list = list(), list(), list(), list()
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
            logits_list.append(logits.cpu())
            preds_list.append(preds.cpu())
            labels_list.append(y)
        return torch.cat(feats_list, dim=0), torch.cat(logits_list, dim=0), torch.cat(preds_list, dim=0), torch.cat(labels_list, dim=0)

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

    def select_bridge_pairs(self, net, id_data_loader, W, R, feature_mean):
        """
        Select top-N bridge endpoints per class pair (same logic as BBR).
        
        Returns:
            bridge_pairs_per_class_pair: dict {class_pair, bridge_pairs};
                bridge_pairs is a list of (u, v) bridge endpoint tensors
        """
        features, logits, preds, labels = self.total_inference(net, id_data_loader)
        labels = labels.numpy()
        classes = np.unique(labels)
        correct_mask = (preds.numpy() == labels)

        bridge_pairs_per_class_pair = dict()

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                cls_i_mask = (labels == classes[i]) & correct_mask
                cls_j_mask = (labels == classes[j]) & correct_mask

                feats_i = features[cls_i_mask].cuda()
                feats_j = features[cls_j_mask].cuda()

                if feats_i.shape[0] == 0 or feats_j.shape[0] == 0:
                    continue

                # Compute distances (whitened or not)
                # if self.whiten_features and self.whitened_top_n_selection:
                if self.whitened_top_n_selection:
                    feats_i_w = self.whiten(feats_i, W.cuda(), feature_mean.cuda())
                    feats_j_w = self.whiten(feats_j, W.cuda(), feature_mean.cuda())
                    dists = torch.cdist(feats_i_w, feats_j_w, p=2)
                # elif self.whiten_features and not self.whitened_top_n_selection:
                #     dists = torch.cdist(feats_i, feats_j, p=2)
                else:
                    dists = torch.cdist(feats_i, feats_j, p=2)

                nA, nB = feats_i.shape[0], feats_j.shape[0]
                flat_dists = dists.view(-1)
                _, top_indices = torch.topk(flat_dists, k=min(self.top_n, nA * nB), largest=False)
                top_pairs = [(idx // nB, idx % nB) for idx in top_indices.cpu().tolist()]

                bridge_pairs = []
                for iA, iB in top_pairs:
                    u = feats_i[iA:iA+1]
                    v = feats_j[iB:iB+1]
                    bridge_pairs.append((u, v))

                bridge_pairs_per_class_pair[(classes[i], classes[j])] = bridge_pairs

        return bridge_pairs_per_class_pair