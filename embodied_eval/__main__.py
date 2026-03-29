import argparse
import os
import sys

from loguru import logger as eval_logger

from embodied_eval.evaluators import build_evaluator

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,)
    parser.add_argument(
        "--model_args",
        default="",
        help="model_name_or_path=,lora_id=",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="eqa",
        help="Choose Evaluator: eqa or nav. "
    )
    parser.add_argument(
        "--tasks",
        default=None,
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Environment name for navigation tasks"
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    parser.add_argument(
        "--inference_only",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save_results",
        default=True,
        action="store_true",
        help="Save evaluation results to file"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="DEBUG",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode"
    )
    parser.add_argument(
        "--gpu_memory_utilization", 
        type=float, 
        default=0.8, 
        help="GPU memory utilization (0.0-1.0)"
    )
    return parser.parse_args()


def cli_evaluate(args):
    # reset logger
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    # initialize Evaluator
    evaluator = build_evaluator(args)

    try:
        # inference
        eval_logger.info("Starting inference...")
        inference_results = evaluator.inference()
        
        if args.inference_only:
            eval_logger.info("Inference completed, saving inference results...")
            if args.save_results:
                evaluator.save_inference_results(inference_results)
            eval_logger.info("Inference results saved, exiting without evaluation.")
            return
        else:
            # evaluation
            eval_logger.info("Inference completed, proceeding to evaluation.")
            results_dict = evaluator.evaluate(inference_results)
            evaluator.print_results(results_dict)
            
            if args.save_results:
                eval_logger.info("Saving evaluation results...")
                evaluator.save_results(results_dict)

    except Exception as e:
        eval_logger.error(f"Error during evaluation: {e}.")

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 20092))
        debugpy.wait_for_client()
        eval_logger.info("Waiting for debugger to attach...")
    cli_evaluate(args)