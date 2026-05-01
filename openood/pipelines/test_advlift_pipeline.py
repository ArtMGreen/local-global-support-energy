from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger

import numpy as np


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
        evaluator = get_evaluator(self.config)

        # init postprocessor
        postprocessor = get_postprocessor(self.config)

        print('\nStart evaluation...', flush=True)
        test_metrics = evaluator.eval_advlift(net, test_loader, postprocessor)

        advlift_list = test_metrics['advlift']
        percentiles = [25, 50, 75]
        q25, q50, q75 = np.percentile(advlift_list, percentiles)
        advlift_mean = float(np.mean(advlift_list))
        
        fr = test_metrics['fr']
        asr = test_metrics['asr']
        clean_acc = test_metrics['clean_acc']
        robust_acc = test_metrics['robust_acc']

        print(f"\nEvaluation complete!\nAdvLift: mean={advlift_mean}, Q1={q25:.5f}, median={q50:.5f}, Q3={q75:.5f}, ASR: {(100*asr):.2f}%, FR: {(100*fr):.2f}%, ACC (clean): {clean_acc:.5f}, ACC (robust): {robust_acc:.5f}", flush=True)

        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Histogram
        ax1.hist(advlift_list, bins='auto', edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quantile plot
        sorted_data = np.sort(advlift_list)
        p = np.linspace(0, 100, len(sorted_data))  # percentages from 0 to 100
        
        ax2.plot(p, sorted_data, 'b-', linewidth=1.5)
        
        # Mark and label the quantiles
        ax2.axhline(y=q25, color='r', linestyle='--', alpha=0.2)
        ax2.axhline(y=q50, color='g', linestyle='--', alpha=0.2)
        ax2.axhline(y=q75, color='r', linestyle='--', alpha=0.2)
        
        ax2.axvline(x=25, color='r', linestyle='--', alpha=0.2)
        ax2.axvline(x=50, color='g', linestyle='--', alpha=0.2)
        ax2.axvline(x=75, color='r', linestyle='--', alpha=0.2)

        # Add a text box with all statistics
        stats_text = f'Q1 (25th): {q25:.4f}\nMedian (50th): {q50:.4f}\nQ3 (75th): {q75:.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add text labels with actual numbers
        ax2.text(1, q25, f'Q1={q25:.4f}', ha='left', va='bottom', fontsize=9, color='red')
        ax2.text(1, q50, f'Median={q50:.4f}', ha='left', va='bottom', fontsize=9, color='green', fontweight='bold')
        ax2.text(1, q75, f'Q3={q75:.4f}', ha='left', va='bottom', fontsize=9, color='red')
        
        # Labels and title
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Value')
        ax2.set_title('Quantile Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Show the plot
        # plt.show()
        plt.savefig(f'{self.config.output_dir}/{self.config.mark}.png', dpi=300, bbox_inches='tight')
