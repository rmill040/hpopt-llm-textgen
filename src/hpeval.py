"""Hyperparameter optimization evaluation."""
import argparse
import json
import logging
import time
from typing import Any, Dict, Union

import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import create_inference_dataset, free_memory, set_seed

BEST_METRIC = -1 * float("inf")
ROUGE = evaluate.load("rouge")
BERTSCORE = evaluate.load("bertscore")
DATALOADER = None
MODEL = None
TOKENIZER = None

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


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to results")
    parser.add_argument("--sample-size", required=True, type=sample_size_type, help="Sample size of dataset")

    args = parser.parse_args()
    return args


def batch_inference(*, generation_type: str, metric: str, params: Dict[str, Any], random_state: int) -> Dict[str, Any]:
    """Batch inference function."""
    global BEST_METRIC
    set_seed(random_state)

    LOGGER.info(f"Evaluating {generation_type} method: {params}")

    results = {}
    try:
        tic = time.time()

        # Compute predictions
        predictions_all = []
        references_all = []
        n_batches = len(DATALOADER)
        for idx, batch in tqdm(enumerate(DATALOADER, 1), position=1, desc="batch", leave=False):
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
            device="cuda:7",
        )
        free_memory()

        results = {
            **rouge_metrics,
            "bertscore_f1": np.mean(bertscore_metrics["f1"]),
            "bertscore_precision": np.mean(bertscore_metrics["precision"]),
            "bertscore_recall": np.mean(bertscore_metrics["recall"]),
            "total_time": toc - tic,
        }

        if results[metric] > BEST_METRIC:
            BEST_METRIC = results[metric]
            LOGGER.info(f"New best {metric} metric found: {results[metric]} | {generation_type}")
        else:
            LOGGER.info(f"Results: {results[metric]} | {generation_type}")

    except Exception as e:
        LOGGER.info(f"Failed to run batch inference because: {e}")

    return results


def main():
    """Main program."""
    global DATALOADER, MODEL, TOKENIZER

    # Parse cli
    args = cli()

    # Load hp opt results
    with open(args.path) as f:
        results = json.load(f)
    config = results["config"]

    # Load model/tokenizer
    LOGGER.info("Loading pretrained tokenizer and model")
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        f"{config['model_name']}/model",
        device_map="balanced_low_0",
        torch_dtype=torch.float16,
        use_cache=True,
        load_in_8bit=True,
        quantization_config=quantization_config,
    )
    MODEL.eval()
    TOKENIZER = AutoTokenizer.from_pretrained(f"{config['model_name']}/model", padding_side="left")

    # Create dataset
    LOGGER.info("Creating dataset")
    DATALOADER = create_inference_dataset(
        tokenizer=TOKENIZER,
        dataset_name=config["dataset_name"],
        split="test",
        batch_size=config["batch_size"],
        sample_size=args.sample_size,
        max_length=1024,
        random_state=config["random_state"],
    )
    # Determine which params to test
    df_validation = pd.DataFrame(results["validation"])
    df_validation["loss"] = -1 * df_validation["loss"]  # Convert to positive metric

    # Take top scoring params for each method
    idx = df_validation.groupby("type")["loss"].idxmax().values
    selected = df_validation.iloc[idx].sort_values("loss", ascending=False).to_dict(orient="records")

    # Run batch inference
    final_results = []
    for row in tqdm(selected, position=0, desc="hp config"):
        current = {
            "type": row["type"],
            "params": row["params"],
            "validation": row["metrics"],
            "metric": config["metric"],
        }
        current["test"] = batch_inference(
            generation_type=current["type"],
            metric=current["metric"],
            params=current["params"],
            random_state=config["random_state"],
        )
        final_results.append(current)

    # Update config
    sample_size_validation = config.pop("sample_size")
    config["sample_size_validation"] = sample_size_validation
    config["sample_size_test"] = args.sample_size
    final_results = [config] + final_results

    # Save
    path = args.path.replace("hpopt", "hpeval")
    with open(path, "w") as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    main()
