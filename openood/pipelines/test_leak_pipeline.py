import numpy as np
import torch
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestLeakPipeline:
    def __init__(self, config) -> None:
        self.config = config
        postproc_args = config.postprocessor.postprocessor_args
        self.tau = postproc_args.tau
        # self.whiten_features = config.evaluator.evaluator_args.whiten_features
        self.whitened_top_n_selection = config.evaluator.evaluator_args.whitened_top_n_selection
        self.eta_id_coverage = config.evaluator.evaluator_args.eta_id_coverage_fraction
        self.max_steps = postproc_args.max_steps

    def compute_energy_scores(self, net, dataloader):
        _, logits, _, _ = self.evaluator.total_inference(net, dataloader)
        T = self.config.postprocessor.postprocessor_args.temperature
        energy = -T * torch.logsumexp(logits / T, dim=1)
        return energy.numpy()

    def compute_eta(self, id_val_energies, ood_val_energies=None):
        self.eta = np.quantile(id_val_energies, self.eta_id_coverage)
        self.actual_id_coverage = np.mean(id_val_energies < self.eta)
        print(f"Achieved ID coverage at eta={self.eta:.4f}: {self.actual_id_coverage:.4f} (target: {self.eta_id_coverage})")
        if ood_val_energies is not None:
            self.ood_rejected = np.mean(ood_val_energies > self.eta)
            print(f"OOD rejection at eta: {(100 * self.ood_rejected):.2f}%")
        else:
            self.ood_rejected = None
        return self.eta, self.actual_id_coverage, self.ood_rejected

    def compute_leak_k(self, energy_histories_per_pair, eta):
        """
        Compute Leak-K: fraction of points below eta after K steps, for K=0..max_steps.
        
        Args:
            energy_histories_per_pair: dict {class_pair tuples, (max_steps+1, pts_per_class_pair) arrays}
            eta: energy threshold
        
        Returns:
            leak_overall: (max_steps+1,) array — overall fraction below eta
            leak_per_class_pair: dict {class_pair, (max_steps+1,) arrays}
        """
        all_histories = np.concatenate(list(energy_histories_per_pair.values()), axis=1)  # (max_steps+1, pts_per_bridge*top_n)

        leak_overall = np.mean(np.minimum.accumulate(all_histories, axis=0) < eta, axis=1)  # (max_steps+1,)
        
        leak_per_class_pair = {
            pair: np.mean(np.minimum.accumulate(hist, axis=0) < eta, axis=1) 
            for pair, hist in energy_histories_per_pair.items()
        }
        
        return leak_overall, leak_per_class_pair

    def run(self):
        setup_logger(self.config)

        # Get dataloaders
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        train_loader = id_loader_dict['train']
        test_loader = id_loader_dict['test']
        id_val_loader = id_loader_dict['val']
        ood_val_loader = ood_loader_dict['val']

        # Init network
        net = get_network(self.config.network)

        # Init evaluator
        self.evaluator = get_evaluator(self.config)

        # Init postprocessor
        postprocessor = get_postprocessor(self.config)

        # Compute whitening if needed
        # if self.whiten_features:
        if self.whitened_top_n_selection:
            print("Computing whitening matrix...")
            train_features, train_logits, train_preds, train_labels = self.evaluator.total_inference(net, train_loader)
            W, R, mean_feature = self.evaluator.compute_whitening(train_features)
        else:
            W, R, mean_feature = None, None, None

        print("\nComputing ID vs OOD threshold eta...")
        id_val_energies = self.compute_energy_scores(net, id_val_loader)
        ood_val_energies = self.compute_energy_scores(net, ood_val_loader)
        self.compute_eta(id_val_energies, ood_val_energies)
        
        print("\nSelecting bridge pairs...")
        bridge_pairs_per_class_pair = self.evaluator.select_bridge_pairs(
            net, test_loader, W, R, mean_feature
        )

        print(f"\nRunning Leak inference on {len(bridge_pairs_per_class_pair)} class pairs...")
        energy_histories_per_pair = postprocessor.inference(
            net, bridge_pairs_per_class_pair, progress=True
        )
        
        leak_overall, leak_per_class_pair = self.compute_leak_k(energy_histories_per_pair, self.eta)

        print("\n" + "="*60)
        print(f"Leak Evaluation Results (eta={self.eta:.4f})")
        print("="*60)
        print(f"{'Step':>5s} | {'Leak (overall)':>15s}")
        print("-"*25)
        for k in range(min(self.max_steps + 1, 11)):  # Print first 10 steps + final
            print(f"{k:5d} | {leak_overall[k]:15.4f}")
        if self.max_steps > 10:
            print(f"...  | ...")
            print(f"{self.max_steps:5d} | {leak_overall[-1]:15.4f}")
        
        print(f"\nFinal Leak (K={self.max_steps}): {leak_overall[-1]:.4f}")
        print(f"Initial Leak (K=0): {leak_overall[0]:.4f}")
        print(f"Leak deterioration: {leak_overall[-1] - leak_overall[0]:.4f}")

        self.save_results(leak_overall, leak_per_class_pair, energy_histories_per_pair)

    def save_results(self, leak_overall, leak_per_class_pair, energy_histories_per_pair):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df_overall = pd.DataFrame({
            'k': np.arange(len(leak_overall)),
            'leak': leak_overall
        })
        df_overall["tau"] = self.tau
        df_overall["lr"] = self.config.postprocessor.postprocessor_args.lr
        df_overall["eta"] = self.eta
        df_overall["target_id_coverage"] = self.eta_id_coverage
        df_overall["achieved_id_coverage"] = self.actual_id_coverage
        df_overall["ood_rejection"] = self.ood_rejected
        df_overall["pts_per_bridge"] = self.config.postprocessor.postprocessor_args.pts_per_bridge
        df_overall["top_n"] = self.config.evaluator.evaluator_args.top_n
        df_overall["seed"] = self.config.seed
        df_overall["model"] = self.config.network.name
        df_overall["checkpoint"] = self.config.network.checkpoint
        df_overall["dataset"] = self.config.dataset.name
        
        df_overall.to_csv(self.config.output_dir + "/leak.csv", index=False)
        
        rows = list()
        for (ci, cj), leak_cp in leak_per_class_pair.items():
            for step, leak_val in enumerate(leak_cp):
                rows.append({
                    'class_i': ci,
                    'class_j': cj,
                    'k': step,
                    'leak': leak_val
                })
        df_pairs = pd.DataFrame(rows)
        df_pairs["tau"] = self.tau
        df_pairs["lr"] = self.config.postprocessor.postprocessor_args.lr
        df_pairs["eta"] = self.eta
        df_pairs["target_id_coverage"] = self.eta_id_coverage
        df_pairs["achieved_id_coverage"] = self.actual_id_coverage
        df_pairs["ood_rejection"] = self.ood_rejected
        df_pairs["pts_per_bridge"] = self.config.postprocessor.postprocessor_args.pts_per_bridge
        df_pairs["top_n"] = self.config.evaluator.evaluator_args.top_n
        df_pairs["seed"] = self.config.seed
        df_pairs["model"] = self.config.network.name
        df_pairs["checkpoint"] = self.config.network.checkpoint
        df_pairs["dataset"] = self.config.dataset.name
        df_pairs.to_csv(self.config.output_dir + "/leak_per_class_pair.csv", index=False)
        
        plot_dir = self.config.output_dir# + "/energy_plots"
        # plot_dir.mkdir(exist_ok=True)
        
        all_histories = np.concatenate(list(energy_histories_per_pair.values()), axis=1)
        max_steps = all_histories.shape[0] - 1
        
        for step in range(max_steps + 1):
            # Overall plot for this step
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(all_histories[step], bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=self.eta, color='red', linestyle='--', linewidth=2, label=f'eta={self.eta:.3f}')
            ax.set_title(f'Overall Energy Distribution — Step {step}')
            ax.set_xlabel('Energy')
            ax.set_ylabel('Count')
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir + f"/{self.config.mark}_step_{step:03d}.png", dpi=150)
            plt.close(fig)
            
            # Per-class-pair plot for this step
            # for (ci, cj), hist_cp in energy_histories_per_pair.items():
            #     fig, ax = plt.subplots(figsize=(8, 5))
            #     ax.hist(hist_cp[step], bins=min(50, hist_cp.shape[1] // 10), alpha=0.7, edgecolor='black')
            #     ax.axvline(x=self.eta, color='red', linestyle='--', linewidth=2, label=f'eta={self.eta:.3f}')
            #     ax.set_title(f'Energy Distribution — Class {ci} vs Class {cj}, Step {step}')
            #     ax.set_xlabel('Energy')
            #     ax.set_ylabel('Count')
            #     ax.legend()
            #     fig.tight_layout()
            #     fig.savefig(plot_dir + f"/class_{ci}_vs_{cj}_step_{step:03d}.png", dpi=150)
            #     plt.close(fig)
        
        print(f"Results saved to {self.config.output_dir}")