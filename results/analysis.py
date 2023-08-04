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


def create_hpopt_metrics_sft_full() -> None:
    """Create hyperparameter optimization metrics for full search spaces with SFT models."""
    # Format metrics
    hp_metrics = {}
    for dataset in DATASETS:
        hp_metrics[dataset] = {}
        for model in MODELS:
            hpopt_metrics = json.load(open(glob(str(HERE / dataset / "hpopt_sft_full" / f"{model}*hpopt.json"))[0]))
            hpeval_metrics = json.load(open(glob(str(HERE / dataset / "hpopt_sft_full" / f"{model}*hpeval.json"))[0]))

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
    df_hp.to_csv(HERE / "hpopt_metrics_sft_full.csv", index=True)

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
    df_hp.to_csv(HERE / "hpeval_metrics_sft_full.csv", index=True)


def create_hpopt_metrics_sft_targeted() -> None:
    """Create hyperparameter optimization metrics for targeted search spaces with SFT models."""
    # Format metrics
    hp_metrics = {}
    for dataset in DATASETS:
        hp_metrics[dataset] = {}
        for strategy in STRATEGIES:
            hpopt_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_sft_targeted" / f"{strategy}-opt*hpopt.json"))[0])
            )
            hpeval_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_sft_targeted" / f"{strategy}-opt*hpeval.json"))[0])
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
    df_hp.to_csv(HERE / "hpopt_metrics_sft_targeted.csv", index=True)

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
    df_hp.to_csv(HERE / "hpeval_metrics_sft_targeted.csv", index=True)


def create_hpopt_metrics_nosft_targeted() -> None:
    """Create hyperparameter optimization metrics for targeted search spaces without SFT models."""
    # Format metrics
    hp_metrics = {}
    for dataset in DATASETS:
        hp_metrics[dataset] = {}
        for strategy in STRATEGIES:
            hpopt_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_nosft_targeted" / f"{strategy}-opt*hpopt.json"))[0])
            )
            hpeval_metrics = json.load(
                open(glob(str(HERE / dataset / "hpopt_nosft_targeted" / f"{strategy}-opt*hpeval.json"))[0])
            )

            # Ensure configs match
            assert hpopt_metrics["config"]["model_name"] == "facebook/opt-1.3b", f"{dataset} - {strategy}"
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
    df_hp.to_csv(HERE / "hpopt_metrics_nosft_targeted.csv", index=True)

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
    df_hp.to_csv(HERE / "hpeval_metrics_nosft_targeted.csv", index=True)


def main() -> None:
    """Create tables and figures for paper."""
    ####################
    # AGGREGATE TABLES #
    ####################

    for path, func in [
        ("sft_metrics.csv", create_sft_metrics),
        ("hpopt_metrics_sft_full.csv", create_hpopt_metrics_sft_full),
        ("hpeval_metrics_sft_full.csv", create_hpopt_metrics_sft_full),
        ("hpopt_metrics_sft_targeted.csv", create_hpopt_metrics_sft_targeted),
        ("hpeval_metrics_sft_targeted.csv", create_hpopt_metrics_sft_targeted),
        ("hpeval_metrics_nosft_targeted", create_hpopt_metrics_nosft_targeted),
        ("hpeval_metrics_nosft_targeted", create_hpopt_metrics_nosft_targeted),
    ]:
        if not os.path.exists(HERE / path):
            func()

    # Create categorical data type so that order of strategies is consistent in tables and figures
    strategies = [s.replace("-", " ").replace("top k", "top-k").replace("top p", "top-p") for s in STRATEGIES]
    labels = [s.replace("top-k", "top-$k$").replace("top-p", "top-$p$") for s in strategies]
    strategies_dtype = CategoricalDtype(strategies, ordered=True)
    models_dtype = CategoricalDtype(
        ["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"], ordered=True
    )

    #################
    # SINGLE TABLES #
    #################

    df_hpeval_nosft_targeted = pd.read_csv("hpeval_metrics_nosft_targeted.csv")
    df_hpeval_sft_targeted = pd.read_csv("hpeval_metrics_sft_targeted.csv")
    df_hpeval_sft_full = pd.read_csv("hpeval_metrics_sft_full.csv")

    # BERTScore F1 metrics
    # OPT-1.3b model results in No SFT + targeted space, SFT + targeted space, and SFT + full space
    for label, df in zip(
        ["No SFT + Targeted", "SFT + Targeted", "SFT + Full"],
        [df_hpeval_nosft_targeted, df_hpeval_sft_targeted, df_hpeval_sft_full],
    ):
        df["type"] = df["type"].astype(strategies_dtype)
        df["test.total_time"] = df["test.total_time"] / df["test.total_time"].sum()
        df = df.sort_values("type").reset_index(drop=True)

        # Select only OPT-1.3b model with the full search space results
        if "model" in df:
            df = df[df["model"] == "opt-1.3b"].reset_index(drop=True).drop("model", axis=1)

        rows = {strategy: {"xsum": [], "cnn_dailymail": [], "samsum": []} for strategy in strategies}
        for group, data in df.groupby("dataset"):
            for _, row in data.iterrows():
                rows[row.type][row.dataset].append(round(row["validation.bertscore_f1"], 2))
                rows[row.type][row.dataset].append(round(row["test.bertscore_f1"], 2))

        # Print for copy/paste into LaTeX
        print("\n", label, "BERTScore F1")
        for key, value in rows.items():
            string = [
                r"& \multicolumn{1}{l}" + "{" + key + "}",
                *list(map(lambda f: format(f, ".2f"), value["xsum"])),
                *list(map(lambda f: format(f, ".2f"), value["cnn_dailymail"])),
                *list(map(lambda f: format(f, ".2f"), value["samsum"])),
            ]
            print(" & ".join(string) + r" \\")

    print("\n" * 5)

    # ROUGE-1 metrics
    # OPT-1.3b model results in No SFT + targeted space, SFT + targeted space, and SFT + full space
    for label, df in zip(
        ["No SFT + Targeted", "SFT + Targeted", "SFT + Full"],
        [df_hpeval_nosft_targeted, df_hpeval_sft_targeted, df_hpeval_sft_full],
    ):
        # Select only OPT-1.3b model with the full search space results
        if "model" in df:
            df = df[df["model"] == "opt-1.3b"].reset_index(drop=True).drop("model", axis=1)

        rows = {strategy: {"xsum": [], "cnn_dailymail": [], "samsum": []} for strategy in strategies}
        for group, data in df.groupby("dataset"):
            for _, row in data.iterrows():
                rows[row.type][row.dataset].append(round(row["validation.rouge1"], 2))
                rows[row.type][row.dataset].append(round(row["test.rouge1"], 2))

        # Print for copy/paste into LaTeX
        print("\n", label, "ROUGE-1")
        for key, value in rows.items():
            string = [
                r"& \multicolumn{1}{l}" + "{" + key + "}",
                *list(map(lambda f: format(f, ".2f"), value["xsum"])),
                *list(map(lambda f: format(f, ".2f"), value["cnn_dailymail"])),
                *list(map(lambda f: format(f, ".2f"), value["samsum"])),
            ]
            print(" & ".join(string) + r" \\")

    print("\n" * 5)
    
    # BERTScore F1 metrics
    # Across all datasets SFT models on full search space
    for group, data in df_hpeval_sft_full.groupby("dataset"):
        rows = {
            strategy: {
                "opt-125m": [],
                "opt-350m": [],
                "opt-1.3b": [],
                "opt-2.7b": [],
                "opt-6.7b": [],
                "opt-13b": [],
                "opt-30b": [],
            }
            for strategy in strategies
        }
        data = data.sort_values("type").reset_index(drop=True)
        for _, row in data.iterrows():
            rows[row.type][row.model].append(round(row["validation.bertscore_f1"], 2))
            rows[row.type][row.model].append(round(row["test.bertscore_f1"], 2))
        
        # Print for copy/paste into LaTeX
        print("\n", group, "BERTScore F1")
        for key, value in rows.items():
            string = [
                key,
                *list(map(lambda f: format(f, ".2f"), value["opt-125m"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-350m"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-1.3b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-2.7b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-6.7b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-13b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-30b"])),
            ]
            print(" & ".join(string) + r" \\")

    print("\n" * 5)

    # ROUGE-1 metrics
    # Across all datasets SFT models on full search space
    for group, data in df_hpeval_sft_full.groupby("dataset"):
        rows = {
            strategy: {
                "opt-125m": [],
                "opt-350m": [],
                "opt-1.3b": [],
                "opt-2.7b": [],
                "opt-6.7b": [],
                "opt-13b": [],
                "opt-30b": [],
            }
            for strategy in strategies
        }
        data = data.sort_values("type").reset_index(drop=True)
        for _, row in data.iterrows():
            rows[row.type][row.model].append(round(row["validation.rouge1"], 2))
            rows[row.type][row.model].append(round(row["test.rouge1"], 2))
        
        # Print for copy/paste into LaTeX
        print("\n", group, "ROUGE-1")
        for key, value in rows.items():
            string = [
                key,
                *list(map(lambda f: format(f, ".2f"), value["opt-125m"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-350m"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-1.3b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-2.7b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-6.7b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-13b"])),
                *list(map(lambda f: format(f, ".2f"), value["opt-30b"])),
            ]
            print(" & ".join(string) + r" \\")

    print("\n" * 5)
    
    ###########
    # FIGURES #
    ###########

    # Boxplot for total compute time used by each strategy in the full space
    ax = sns.boxplot(data=df_hpeval_sft_full, y="type", x="test.total_time", orient="h")
    ax.set_ylabel("")
    ax.set_xlabel("Proportion of Total Inference Time", labelpad=15)
    ax.set_yticklabels(rotation=0, labels=labels)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.tight_layout()
    plt.savefig("sft_full_time.png", dpi=300)
    plt.close()

    df_hpopt_nosft_targeted = pd.read_csv("hpopt_metrics_nosft_targeted.csv")
    df_hpopt_sft_targeted = pd.read_csv("hpopt_metrics_sft_targeted.csv")
    df_hpopt_sft_full = pd.read_csv("hpopt_metrics_sft_full.csv")

    for df in [
        df_hpopt_nosft_targeted,
        df_hpopt_sft_targeted,
        df_hpopt_sft_full,
    ]:
        df["type"] = df["type"].astype(strategies_dtype)

    # BERTScore F1 metrics
    # Hpopt results for full search space for SFT models
    table = df_hpopt_sft_full.groupby(["dataset", "model"])["loss"].describe().reset_index().drop([
        "count", "mean", "std", "25%", "50%", "75%"
    ], axis=1)
    
    idxmax = df_hpopt_sft_full.groupby(["dataset", "model"])["loss"].idxmax()
    table["trial"] = df_hpopt_sft_full.iloc[idxmax]["trial"].values
    table["type"] = df_hpopt_sft_full.iloc[idxmax]["type"].values
    table["model"] = table["model"].astype(models_dtype)
    for group, data in table.groupby("dataset"):
        print("\n", group, "SFT Models", "Full", "BERTScore F1")
        data = data.sort_values(["dataset", "model"]).reset_index(drop=True)
        for _, row in data.iterrows():
            string = [
                row["model"].replace("opt-", "").upper(),
                r"\multicolumn{1}{l}" "{" + row["type"] + "}",
                row["trial"],
                format(row["min"], ".2f"),
                format(row["max"], ".2f"),
            ]
            print("& " + " & ".join(map(str, string)) + r" \\")

    # BERTScore F1 metrics 
    # Hpopt results for targeted search space for no SFT models
    table = df_hpopt_nosft_targeted.groupby(["dataset", "type"])["loss"].describe().reset_index().drop([
        "count", "mean", "std", "25%", "50%", "75%"
    ], axis=1)
    
    idxmax = df_hpopt_nosft_targeted.groupby(["dataset", "type"])["loss"].idxmax()
    table["trial"] = df_hpopt_nosft_targeted.iloc[idxmax]["trial"].values
    for group, data in table.groupby("dataset"):
        print("\n", group, "No SFT Models", "Targeted", "BERTScore F1")
        data = data.sort_values(["dataset", "type"]).reset_index(drop=True)
        for _, row in data.iterrows():
            string = [
                r"\multicolumn{1}{l}" "{" + row["type"] + "}",
                row["trial"],
                format(row["min"], ".2f"),
                format(row["max"], ".2f"),
            ]
            print("& " + " & ".join(map(str, string)) + r" \\")


    # BERTScore F1 metrics 
    # Hpopt results for targeted search space for SFT models
    table = df_hpopt_sft_targeted.groupby(["dataset", "type"])["loss"].describe().reset_index().drop([
        "count", "mean", "std", "25%", "50%", "75%"
    ], axis=1)
    
    idxmax = df_hpopt_sft_targeted.groupby(["dataset", "type"])["loss"].idxmax()
    table["trial"] = df_hpopt_sft_targeted.iloc[idxmax]["trial"].values
    for group, data in table.groupby("dataset"):
        print("\n", group, "SFT Models", "Targeted", "BERTScore F1")
        data = data.sort_values(["dataset", "type"]).reset_index(drop=True)
        for _, row in data.iterrows():
            string = [
                r"\multicolumn{1}{l}" "{" + row["type"] + "}",
                row["trial"],
                format(row["min"], ".2f"),
                format(row["max"], ".2f"),
            ]
            print("& " + " & ".join(map(str, string)) + r" \\")

    import pdb; pdb.set_trace()
    
    
    # Test metrics
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
    # sns.boxplot(data=df, x="type", y="test.bertscore_f1", ax=ax[0])
    # sns.boxplot(data=df, x="type", y="test.bertscore_precision", ax=ax[1])
    # sns.boxplot(data=df, x="type", y="test.bertscore_recall", ax=ax[2])
    # ax[0].set_xticklabels(rotation=90, labels=labels)
    # ax[1].set_xticklabels(rotation=90, labels=labels)
    # ax[2].set_xticklabels(rotation=90, labels=labels)
    # plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    # sns.boxplot(data=df, x="type", y="test.rouge1", ax=ax[0])
    # sns.boxplot(data=df, x="type", y="test.rouge2", ax=ax[1])
    # sns.boxplot(data=df, x="type", y="test.rougeL", ax=ax[2])
    # ax[0].set_xticklabels(rotation=90, labels=labels)
    # ax[1].set_xticklabels(rotation=90, labels=labels)
    # ax[2].set_xticklabels(rotation=90, labels=labels)
    # plt.show()

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
