from evals.base import Evaluator

class tripunlambEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("tripunlamb", eval_cfg, **kwargs)