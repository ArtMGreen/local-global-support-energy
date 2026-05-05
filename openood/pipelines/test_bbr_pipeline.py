from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
# from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestBBRPipeline:
    def __init__(self, config) -> None:
        self.config = config
        self.tau = self.config.evaluator.evaluator_args.tau

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
        # postprocessor = get_postprocessor(self.config)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        test_metrics = evaluator.eval_bbr(net, test_loader)
        print('\nEvaluation complete (tau={}), BBR={:.2f}'.format(self.tau, test_metrics['bbr']), flush=True)

        import matplotlib.pyplot as plt
        import numpy as np
        
        curves_dict = test_metrics['curves_by_pairs']
        
        plt.figure(figsize=(10, 10))
        
        for class_nums, (x, y) in curves_dict.items():
            plt.plot(x, y, alpha=0.7) #, label='Classes {} vs {}'.format(*class_nums))
        
        plt.xlabel('tau')
        plt.ylabel('energy')
        plt.title('Energy bridges')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside
        plt.grid(True, alpha=0.25)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Show the plot
        # plt.show()
        plt.savefig(f'{self.config.output_dir}/energy_levels_on_bridges.png', dpi=300, bbox_inches='tight')
