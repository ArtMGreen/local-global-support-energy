from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger

import numpy as np
import pandas as pd


class TestAdvLiftPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init evaluator
        self.evaluator = get_evaluator(self.config)

        # init postprocessor
        self.postprocessor = get_postprocessor(self.config)

        print('\nStart evaluation...', flush=True)
        groupwise_advlift, classwise_advlift = self.evaluator.eval_advlift(net, test_loader, self.postprocessor)
        print("Evaluation complete! Plotting and writing results...", flush=True)

        group_df = self.generate_df(groupwise_advlift)
        class_df = self.generate_df(classwise_advlift)
        result_df = pd.concat([group_df, class_df], ignore_index=True)
        
        # Write to CSV
        save_dir = self.config.output_dir
        csv_path = save_dir + "/aggregated_results.csv"
        result_df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {csv_path}")

        groupwise_save_path = f'{self.config.output_dir}/{self.config.mark}_groupwise.png'
        classwise_save_path = f'{self.config.output_dir}/{self.config.mark}_classwise.png'
        self.plot_sectionwise_results(groupwise_advlift, groupwise_save_path, keys=[["overall"], ["clean-correct", "clean-wrong"], ["attack-success", "attack-fail"], ["flip-success", "flip-fail"]])
        self.plot_sectionwise_results(classwise_advlift, classwise_save_path)

    def generate_df(self, sectionwise_metrics):
        rows = list()
        for section_name, section_dict in sectionwise_metrics.items():
            advlift_list = section_dict["advlift"]
            section_rate = section_dict["rate"]
            rate_description = section_dict["rate_description"]

            percentiles = [25, 50, 75]
            if len(advlift_list) != 0:
                q25, q50, q75 = np.percentile(advlift_list, percentiles)
                advlift_mean = float(np.mean(advlift_list))
            else:
                q25, q50, q75 = None, None, None
                advlift_mean = None

            results_row = {
                "group": section_name,
                "advlift_mean": advlift_mean,
                "advlift_Q1": q25,
                "advlift_median": q50,
                "advlift_Q3": q75,
                "rate": section_rate,
                "rate_description": rate_description
            }
        
            # Add config parameters
            common_args = self.config.adversary.common_args
            pgd_args = self.config.adversary.pgd_args
            
            results_row["adversary"] = self.config.adversary.name
            results_row["attacked_label"] = common_args.label_source
            results_row["eps"] = common_args.eps
            if self.config.adversary.name == "pgd":
                results_row["step_size"] = pgd_args.step_size
                results_row["steps"] = pgd_args.steps
                results_row["random_start"] = pgd_args.random_start
            else:
                results_row["step_size"] = None
                results_row["steps"] = None
                results_row["random_start"] = None
            results_row["seed"] = self.config.seed
            results_row["model"] = self.config.network.name
            results_row["checkpoint"] = self.config.network.checkpoint
            results_row["dataset"] = self.config.dataset.name
            # this one probably will be convenient to sort by
            results_row["Adversary"] = self.postprocessor.adversary.description()

            rows.append(results_row)
        return pd.DataFrame(rows)
    
    def plot_sectionwise_results(self, sectionwise_metrics, save_path, keys=None, figsize=(12, 12)):
        import matplotlib.pyplot as plt
        import numpy as np
        
        if keys is None:
            keys = [[key] for key in sectionwise_metrics.keys()]
        
        n_rows = len(keys)
        fig, axes = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows / 4), sharex=True)
        if n_rows == 1:
            axes = [axes]
        
        for row_idx, key_group in enumerate(keys):
            ax = axes[row_idx]
            
            for col_idx, group_name in enumerate(key_group):
                if group_name not in sectionwise_metrics:
                    continue
                metrics = sectionwise_metrics[group_name]
                advlift = metrics["advlift"]
                if len(advlift) == 0:
                    continue
                    
                color = plt.cm.tab10(col_idx % 10)
                mean_val, q1, median_val, q3 = np.mean(advlift), np.percentile(advlift, 25), np.median(advlift), np.percentile(advlift, 75)
                label = f"{group_name} ({metrics['rate_description']}={metrics['rate']:.3f})"
                
                ax.hist(advlift, bins=50, density=True, alpha=0.3 if len(key_group) > 1 else 0.6, color=color, edgecolor='white', label=label)
                
                vline_color = color if len(key_group) > 1 else 'black'
                for val, ls, lw in [(mean_val, '-', 2), (q1, '--', 1.5), (median_val, ':', 2), (q3, '--', 1.5)]:
                    ax.axvline(val, color=vline_color, linestyle=ls, linewidth=lw, alpha=0.6)
            
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('AdvLift')
        fig.suptitle(f"AdvLift | {self.postprocessor.adversary.description()} | Attacked labels: {self.config.adversary.common_args.label_source}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')