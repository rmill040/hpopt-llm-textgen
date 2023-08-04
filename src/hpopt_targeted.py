"""Hyperparameter optimization for targeted search spaces."""
import argparse
import json
import logging
import time
from math import log
from typing import Any, Dict, Union

import evaluate
import numpy as np
import torch
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import METRICS_DEVICE, create_inference_dataset, free_memory, set_seed

BEST_METRIC = float("inf")
ROUGE = evaluate.load("rouge")
BERTSCORE = evaluate.load("bertscore")
DATALOADER = None
MODEL = None
TOKENIZER = None
CAST_TO_INT = [
    "num_beams",
    "num_beam_groups",
    "top_k",
    "max_new_tokens",
]

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def sample_size_type(value: Union[int, float]) -> Union[int, float]:
    """Type for argparse."""
    value = float(value)
    if value.is_integer() and value != 1:
        value = int(value)
    return value


def metric_types(value: str) -> str:
    """Metric type for optimization."""
    assert value in [
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
    ]
    return value


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, help="Model name")
    parser.add_argument("--dataset-name", required=True, type=str, help="Dataset name")
    parser.add_argument("--batch-size", required=True, type=int, help="Batch size for generating summaries")
    parser.add_argument("--sample-size", required=True, type=sample_size_type, help="Sample size of dataset")
    parser.add_argument("--metric", required=True, type=metric_types, help="Optimization metric")
    parser.add_argument("--max-evals", required=True, type=int, help="Maximum optimization steps")
    parser.add_argument("--random-state", required=True, type=int, help="Random seed to control reproducibility")

    args = parser.parse_args()
    return args


def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """Objective function."""
    global BEST_METRIC

    # Cast dtypes
    for key, value in params.items():
        if key in CAST_TO_INT:
            params[key] = int(value)

    metric = params.pop("metric")
    generation_type = params.pop("type")
    random_state = params.pop("random_state")
    set_seed(random_state)

    results = {
        "loss": None,
        "status": None,
        "message": "",
        "type": generation_type,
        "params": params,
        "metrics": {},
        "total_time": -1,
    }

    LOGGER.info(f"Evaluating {generation_type} method: {params}")

    try:
        tic = time.time()

        # Compute predictions
        predictions_all = []
        references_all = []
        n_batches = len(DATALOADER)
        for idx, batch in enumerate(DATALOADER, 1):
            LOGGER.info(f"Processing batch {idx}/{n_batches}")
            text = batch.pop("text")
            true_summaries = batch.pop("summary")
            batch["input_ids"] = batch["input_ids"].to(0)
            batch["attention_mask"] = batch["attention_mask"].to(0)
            with torch.no_grad():
                predicted_summaries = TOKENIZER.batch_decode(
                    MODEL.generate(**batch, **params),
                    skip_special_tokens=True,
                )
            free_memory()

            for input_text, predicted_summary, true_summary in zip(text, predicted_summaries, true_summaries):
                predictions_all.append(predicted_summary[len(input_text) :])
                references_all.append(true_summary)

        toc = time.time()

        # Compute metrics
        rouge_metrics = ROUGE.compute(predictions=predictions_all, references=references_all)
        bertscore_metrics = BERTSCORE.compute(
            predictions=predictions_all,
            references=references_all,
            idf=True,
            use_fast_tokenizer=True,
            lang="en",
            device=f"cuda:{METRICS_DEVICE}",
        )
        free_memory()

        metrics = {
            **rouge_metrics,
            "bertscore_f1": np.mean(bertscore_metrics["f1"]),
            "bertscore_precision": np.mean(bertscore_metrics["precision"]),
            "bertscore_recall": np.mean(bertscore_metrics["recall"]),
        }
        results["total_time"] = toc - tic
        results["metrics"] = metrics
        results["loss"] = -1 * metrics[metric]
        results["status"] = STATUS_OK

        if results["loss"] < BEST_METRIC:
            BEST_METRIC = results["loss"]
            LOGGER.info(f"New best {metric} metric found: {results['loss']} | {results['type']}")
        else:
            LOGGER.info(f"Results: {results['loss']} | {results['type']}")
    except Exception as e:
        LOGGER.info(f"Failed to run optimization step because: {e}")
        results["status"] = STATUS_FAIL
        results["message"] = str(e)

    return results


def main():
    """Main program."""
    global DATALOADER, MODEL, TOKENIZER

    # Parse cli
    args = cli()

    # Load model/tokenizer
    LOGGER.info("Loading pretrained tokenizer and model")
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        f"{args.model_name}/model" if "facebook" not in args.model_name else args.model_name,
        device_map="balanced_low_0",
        torch_dtype=torch.float16,
        use_cache=True,
        load_in_8bit=True,
        quantization_config=quantization_config,
    )
    MODEL.eval()
    TOKENIZER = AutoTokenizer.from_pretrained(
        f"{args.model_name}/model" if "facebook" not in args.model_name else args.model_name,
        padding_side="left",
    )

    # Create dataset
    LOGGER.info("Creating dataset")
    DATALOADER = create_inference_dataset(
        tokenizer=TOKENIZER,
        dataset_name=args.dataset_name,
        split="validation",
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_length=1024,
        random_state=args.random_state,
    )

    common_hps = {
        "pad_token_id": TOKENIZER.pad_token_id,
        "eos_token_id": TOKENIZER.eos_token_id,
        "bos_token_id": TOKENIZER.bos_token_id,
        "metric": args.metric,
        "max_new_tokens": None,
        "no_repeat_ngram_size": None,
        "repetition_penalty": None,
        "random_state": args.random_state,
    }

    # Hyperparameters that are common across at least 2 spaces
    hp_dists = {
        "no_repeat_ngram_size": {
            "func": hp.pchoice,
            "args": [(0.25, 0), (0.25, 3), (0.25, 4), (0.25, 5)],
        },
        "repetition_penalty": {
            "func": hp.pchoice,
            "args": [(0.5, 1.0), (0.5, 1.2)],
        },
        "max_new_tokens": {
            "func": hp.quniform,
            "args": (50, 150, 1),
        },
        "num_beams": {
            "func": hp.quniform,
            "args": (2, 5, 1),
        },
        "early_stopping": {
            "func": hp.pchoice,
            "args": [(1 / 3, True), (1 / 3, False), (1 / 3, "never")],
        },
        "temperature": {
            "func": hp.loguniform,
            "args": (log(0.6), log(1.2)),
        },
        "top_p": {
            "func": hp.loguniform,
            "args": (log(0.85), log(0.95)),
        },
        "top_k": {
            "func": hp.quniform,
            "args": (40, 60, 1),
        },
    }

    # Create sampling space
    space = [
        {
            "type": "greedy decoding",
            **common_hps,
        },
        {
            "type": "multinomial sampling",
            "do_sample": True,
            "top_k": 0,
            "temperature": None,
            **common_hps,
        },
        {
            "type": "beam search",
            "num_beams": None,
            "early_stopping": None,
            **common_hps,
        },
        {
            "type": "beam search multinomial sampling",
            "top_k": 0,
            "do_sample": True,
            "num_beams": None,
            "early_stopping": None,
            "temperature": None,
            **common_hps,
        },
        {
            "type": "beam search top-k sampling",
            "do_sample": True,
            "num_beams": None,
            "early_stopping": None,
            "temperature": None,
            "top_k": None,
            **common_hps,
        },
        {
            "type": "grouped beam search",
            "num_beams": 4,
            "num_beam_groups": hp.pchoice("num_beam_groups", [(0.5, 2), (0.5, 4)]),
            "diversity_penalty": hp.pchoice("diversity_penalty", [(0.25, 0.0), (0.25, 0.2), (0.25, 0.6), (0.25, 0.8)]),
            "early_stopping": None,
            **common_hps,
        },
        {
            "type": "top-k sampling",
            "do_sample": True,
            "top_k": None,
            "temperature": None,
            **common_hps,
        },
        {
            "type": "top-p nucleus multinomial sampling",
            "do_sample": True,
            "top_k": 0,
            "top_p": None,
            "temperature": None,
            **common_hps,
        },
        {
            "type": "top-p nucleus top-k sampling",
            "do_sample": True,
            "top_p": None,
            "top_k": None,
            "temperature": None,
            **common_hps,
        },
        {
            "type": "contrastive search",
            "penalty_alpha": hp.loguniform("penalty_alpha", log(0.1), log(0.9)),
            "top_k": hp.quniform("top_k contrastive search", 2, 11, q=1),
            "temperature": None,
            **common_hps,
        },
        {
            "type": "truncation multinomial sampling",
            "do_sample": True,
            "top_k": 0,
            "epsilon_cutoff": hp.loguniform("epsilon_cutoff", log(3e-4), log(9e-4)),
            "eta_cutoff": hp.loguniform("eta_cutoff", log(3e-4), log(2e-3)),
            "temperature": None,
            **common_hps,
        },
    ]

    # Update base space
    for config in space:
        for key, value in config.items():
            if config[key] is None:
                hp_dist = hp_dists[key]
                name = f"{key}: {config['type']}"
                config[key] = (
                    hp_dist["func"](name, hp_dist["args"])
                    if hp_dist["func"] == hp.pchoice
                    else hp_dist["func"](name, *hp_dist["args"])
                )

    # Optimize over each search space separately
    for s in space:
        strategy = s["type"].replace(" ", "-")

        # Run optimization
        trials = Trials()
        rstate = np.random.default_rng(args.random_state)
        _ = fmin(objective, space=s, algo=tpe.suggest, max_evals=args.max_evals, trials=trials, rstate=rstate)

        # Save results
        model_name = args.model_name.split("/")[-1]  # Handles cases where we use the base LLM like: 'facebook/opt-1.3b'
        path = (
            f"{strategy}-{model_name}-{args.sample_size}-{args.metric}-{args.max_evals}-{args.random_state}"
            "-hpopt.json"
        )
        results = {
            "config": vars(args),
            "validation": trials.results,
        }

        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
