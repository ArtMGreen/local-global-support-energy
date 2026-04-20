from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


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

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        test_metrics = evaluator.eval_advlift(net, test_loader, postprocessor)
        print('\nComplete Evaluation, AdvLift {:.2f}'.format(test_metrics['advlift']), flush=True)
