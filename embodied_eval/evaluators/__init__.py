from embodied_eval.evaluators.eqa_evaluator import EQAEvaluator
from embodied_eval.evaluators.nav_evaluator import NavEvaluator
from embodied_eval.evaluators.official_beacon3d_evaluator import OfficialBeacon3DEvaluator

def build_evaluator(args):
    """
    Build the evaluator based on the provided arguments.
    """
    if args.evaluator == 'nav':
        return NavEvaluator(args)
    elif args.evaluator == 'official_beacon3d':
        return OfficialBeacon3DEvaluator(args)
    else:
        return EQAEvaluator(args)