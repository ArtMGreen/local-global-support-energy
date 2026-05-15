from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
# from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger

import torch


class TestBBRPipeline:
    def __init__(self, config) -> None:
        self.config = config
        self.tau = self.config.evaluator.evaluator_args.tau
        self.whiten_features = config.evaluator.evaluator_args.whiten_features

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader = loader_dict['train']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init evaluator
        evaluator = get_evaluator(self.config)

        # init postprocessor
        # postprocessor = get_postprocessor(self.config)

        
        if self.whiten_features:
            print("Computing whitening matrix...")
            train_features, train_preds, train_labels = evaluator.total_inference(net, train_loader)
            W, R, mean_feature = evaluator.compute_whitening(train_features)
        else:
            W, R, mean_feature = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        test_metrics = evaluator.eval_bbr(net, test_loader, W, R, mean_feature)
        print('\nEvaluation complete.', flush=True)
