"""Analysis for paper."""
import json
import os
from copy import deepcopy
from glob import glob
from math import exp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype


plt.style.use("ggplot")

HERE = Path(__file__).resolve().parent
DATASETS = ["xsum", "cnn_dailymail", "samsum"]
MODELS = [
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
]
STRATEGIES = [
    "greedy-decoding",
    "multinomial-sampling",
    "beam-search",
    "beam-search-multinomial-sampling",
    "beam-search-top-k-sampling",
    "grouped-beam-search",
    "top-k-sampling",
    "top-p-nucleus-multinomial-sampling",
    "top-p-nucleus-top-k-sampling",
    "contrastive-search",
    "truncation-multinomial-sampling",
]


def create_sft_metrics() -> None:
    """Create supervised fine-tuning results."""
    # Format metrics
    sft_metrics = {}
    for dataset in DATASETS:
        sft_metrics[dataset] = {}
        for model in MODELS:
            best_metrics = json.load(open(HERE / dataset / "sft" / model / "metrics.json"))
            all_metrics = json.load(open(HERE / dataset / "sft" / model / "all_metrics.json"))

            # Since best checkpoint saved, grab the first time that best_metrics["validation_loss"] appears in
            # all_metrics["validation_loss"]
            i = 0
            while True:
                if best_metrics["validation_loss"] == all_metrics[i]["validation_loss"]:
                    break
                else:
                    i += 1

            # Update metrics
            best_metrics["train_epoch"] = all_metrics[i]["epoch"]
            best_metrics["train_step"] = all_metrics[i]["step"]
            best_metrics["train_loss"] = all_metrics[i]["train_loss"]
            best_metrics["train_perplexity"] = exp(all_metrics[i]["train_loss"])
            best_metrics["validation_perplexity"] = all_metrics[i]["perplexity"]
            best_metrics["index"] = i
            sft_metrics[dataset][model] = {
                "best": best_metrics,
                "history": all_metrics,
            }

    df_sft = []
    for dataset in DATASETS:
        for model in MODELS:
            best = deepcopy(sft_metrics[dataset][model]["best"])
            best.pop("index")
            df = pd.DataFrame(best, index=[0])
            df["dataset"] = dataset
            df["model"] = model
            df = df.set_index(["dataset", "model"])
            df_sft.append(df)

    df_sft = pd.concat(df_sft)
    order = [
        "train_epoch",
        "train_step",
        "train_loss",
        "train_perplexity",
        "validation_loss",
        "validation_perplexity",
        "test_loss",
        "test_perplexity",
    ]
    df_sft[order].to_csv(HERE / "sft_metrics.csv", index=True)


def create_hpopt_metrics_full() -> None:
    """Create hyperparameter optimization metrics for full search spaces."""
    # Format metrics
    hp_metrics = {}
    for dataset in DATASETS:
        hp_metrics[dataset] = {}
        for model in MODELS:
            hpopt_metrics = json.load(open(glob(str(HERE / dataset / "hpopt_full" / f"{model}*hpopt.json"))[0]))
            hpeval_metrics = json.load(open(glob(str(HERE / dataset / "hpopt_full" / f"{model}*hpeval.json"))[0]))

            # Ensure configs match
            assert hpopt_metrics["config"]["model_name"] == hpeval_metrics[0]["model_name"], f"{dataset} - {model}"
            assert hpopt_metrics["config"]["dataset_name"] == hpeval_metrics[0]["dataset_name"], f"{dataset} - {model}"

            # Update configs
            hp_metrics[dataset][model] = {
                "config": hpeval_metrics[0],
                "hpopt": hpopt_metrics["validation"],
                "best": hpeval_metrics[1:],
            }

    # Hpopt results
    df_hp = []
    for dataset in DATASETS:
        for model in MODELS:
            hpopt = deepcopy(hp_metrics[dataset][model]["hpopt"])
            for h in hpopt:
                h.pop("status")
                h.pop("message")
                h.pop("params")
            df = pd.json_normalize(hpopt)
            df["loss"] = abs(df["loss"])
            df["dataset"] = dataset
            df["trial"] = range(1, 101)
            df["model"] = model
            df = df.set_index(["dataset", "model", "trial"])
            df_hp.append(df)

    df_hp = pd.concat(df_hp)
    df_hp.to_csv(HERE / "hpopt_metrics_full.csv", index=True)

    # Hpeval results
    df_hp = []
    for dataset in DATASETS:
        for model in MODELS:
            best = deepcopy(hp_metrics[dataset][model]["best"])
            for b in best:
                b.pop("params")
                b.pop("metric")
            df = pd.json_normalize(best)
            df["dataset"] = dataset
            df["model"] = model
            df = df.set_index(["dataset", "model", "type"])
            df_hp.append(df)

    df_hp = pd.concat(df_hp)
    df_hp.to_csv(HERE / "hpeval_metrics_full.csv", index=True)


def create_hpopt_metrics_targeted() -> None:
    """Create hyperparameter optimization metrics for targeted search spaces."""
    # Format metrics
    hp_metrics = {}
    for dataset in DATASETS:
        hp_metrics[dataset] = {}
        for strategy in STRATEGIES:
            hpopt_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_targeted" / f"{strategy}-opt*hpopt.json"))[0])
            )
            hpeval_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_targeted" / f"{strategy}-opt*hpeval.json"))[0])
            )

            # Ensure configs match
            assert hpopt_metrics["config"]["model_name"] == "opt-1.3b", f"{dataset} - {strategy}"
            assert (
                hpopt_metrics["config"]["dataset_name"] == hpeval_metrics[0]["dataset_name"]
            ), f"{dataset} - {strategy}"

            # Update configs
            hp_metrics[dataset][strategy] = {
                "config": hpeval_metrics[0],
                "hpopt": hpopt_metrics["validation"],
                "best": hpeval_metrics[1:],
            }

    # Hpopt results
    df_hp = []
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            hpopt = deepcopy(hp_metrics[dataset][strategy]["hpopt"])
            for h in hpopt:
                h.pop("status")
                h.pop("message")
                h.pop("params")
            df = pd.json_normalize(hpopt)
            df["loss"] = abs(df["loss"])
            df["dataset"] = dataset
            df["trial"] = range(1, 101)
            df["strategy"] = strategy
            df = df.set_index(["dataset", "strategy", "trial"])
            df_hp.append(df)

    df_hp = pd.concat(df_hp)
    df_hp.to_csv(HERE / "hpopt_metrics_targeted.csv", index=True)

    # Hpeval results
    df_hp = []
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            best = deepcopy(hp_metrics[dataset][strategy]["best"])
            for b in best:
                b.pop("params")
                b.pop("metric")
            df = pd.json_normalize(best)
            df["dataset"] = dataset
            df["strategy"] = strategy
            df = df.set_index(["dataset", "strategy", "type"])
            df_hp.append(df)

    df_hp = pd.concat(df_hp)
    df_hp.to_csv(HERE / "hpeval_metrics_targeted.csv", index=True)


def main() -> None:
    """Create tables and figures for paper."""
    ####################
    # AGGREGATE TABLES #
    ####################
    
    for path, func in [
        ("sft_metrics.csv", create_sft_metrics),
        ("hpopt_metrics_full.csv", create_hpopt_metrics_full),
        ("hpeval_metrics_full.csv", create_hpopt_metrics_full),
        ("hpopt_metrics_targeted.csv", create_hpopt_metrics_targeted),
        ("hpeval_metrics_targeted.csv", create_hpopt_metrics_targeted),
    ]:
        if not os.path.exists(HERE / path):
            func()

    ###########
    # FIGURES #
    ###########

    # Create figures
    
    df = pd.read_csv(HERE / "hpopt_metrics_full.csv")

    # Create categorical data type so that figures match order in tables
    strategies = [s.replace("-", " ").replace("top k", "top-k").replace("top p", "top-p") for s in STRATEGIES]
    labels = [s.replace("top-k", "top-$k$").replace("top-p", "top-$p$") for s in strategies]
    strategies_dtype = CategoricalDtype(strategies, ordered=True)
    df["type"] = df["type"].astype(strategies_dtype)

    # # Total time
    # df["test.total_time"] = df["test.total_time"] / df["test.total_time"].sum()
    df = df.sort_values("type").reset_index(drop=True)
    import pdb; pdb.set_trace()
    # ax = sns.boxplot(data=df, x="type", y="test.total_time")
    # ax.set_xlabel("Strategy")
    # ax.set_ylabel("Proportion of Total Time")
    # ax.set_xticklabels(rotation=90, labels=labels)
    # plt.tight_layout()
    # plt.show()
    
    # Test metrics
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
    # sns.boxplot(data=df, x="type", y="test.bertscore_f1", ax=ax[0])
    # sns.boxplot(data=df, x="type", y="test.bertscore_precision", ax=ax[1])
    # sns.boxplot(data=df, x="type", y="test.bertscore_recall", ax=ax[2])
    # ax[0].set_xticklabels(rotation=90, labels=labels)
    # ax[1].set_xticklabels(rotation=90, labels=labels)
    # ax[2].set_xticklabels(rotation=90, labels=labels)
    # plt.show()


    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    sns.boxplot(data=df, x="type", y="test.rouge1", ax=ax[0])
    sns.boxplot(data=df, x="type", y="test.rouge2", ax=ax[1])
    sns.boxplot(data=df, x="type", y="test.rougeL", ax=ax[2])
    ax[0].set_xticklabels(rotation=90, labels=labels)
    ax[1].set_xticklabels(rotation=90, labels=labels)
    ax[2].set_xticklabels(rotation=90, labels=labels)
    plt.show()
    
    
    # plt.tight_layout()
    plt.show()
    import pdb; pdb.set_trace()    
    
    
    
    """                 col1        col2        col3
    bertscores
    
    rouge scores
                                strategy
    """

if __name__ == "__main__":
    main()
