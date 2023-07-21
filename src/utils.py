"""Helper functions."""
import gc
import random
from typing import Any, Dict, Union

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

PROMPT = "Write a summary of the following:\n\n### Document:\n{document}\n\n### Summary:\n{summary}"
CUDA_DEVICES = list(range(torch.cuda.device_count()))
METRICS_DEVICE = CUDA_DEVICES[-1] if len(CUDA_DEVICES) > 1 else 0
EMBEDDINGS_DEVICE = CUDA_DEVICES[-2] if len(CUDA_DEVICES) > 1 else 0


class PromptsInference:
    """Create prompts for inference."""

    @staticmethod
    def samsum(element: Dict[str, Any]) -> Dict[str, Any]:
        """Inference prompts for SAMSum dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document in element["dialogue"]:
            text.append(PROMPT.format(document=document, summary=""))
        return {"text": text}


    @staticmethod
    def cnn_dailymail(element: Dict[str, Any]) -> Dict[str, Any]:
        """Inference prompts for CNN Dailymail dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document in element["article"]:
            text.append(PROMPT.format(document=document, summary=""))
        return {"text": text}

    @staticmethod
    def xsum(element: Dict[str, Any]) -> Dict[str, Any]:
        """Inference prompts for XSum dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document in element["document"]:
            text.append(PROMPT.format(document=document, summary=""))
        return {"text": text}


class PromptsTrain:
    """Create prompts for training."""

    @staticmethod
    def samsum(element: Dict[str, Any]) -> Dict[str, Any]:
        """Training prompts for SAMSum dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document, summary in zip(element["dialogue"], element["summary"]):
            text.append(PROMPT.format(document=document, summary=summary))
        return {"text": text}

    @staticmethod
    def cnn_dailymail(element: Dict[str, Any]) -> Dict[str, Any]:
        """Training prompts for CNN Dailymail dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document, summary in zip(element["article"], element["highlights"]):
            text.append(PROMPT.format(document=document, summary=summary))
        return {"text": text}

    @staticmethod
    def xsum(element: Dict[str, Any]) -> Dict[str, Any]:
        """Training prompts for XSum dataset.

        Parameters
        ----------
        element : Dict[str, Any]
            Input element.

        Returns
        -------
        Dict[str, Any]
            Formatted input.
        """
        text = []
        for document, summary in zip(element["document"], element["summary"]):
            text.append(PROMPT.format(document=document, summary=summary))
        return {"text": text}


def free_memory() -> None:
    """Free memory."""
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_fn_inference(
    element: Dict[str, Any],
    *,
    tokenizer: Any,
    summary_key: str,
    max_length: int,
    add_bos_token: bool,
) -> Dict[str, Any]:
    """Tokenize function for inference.

    Parameters
    ----------
    element : Dict[str, Any]
        Input data.

    tokenizer : Any
        Pretrained tokenizer.

    summary_key : str
        Key for summary column.

    max_length : int
        Maximum sequence length.

    add_bos_token : bool
        Whether to prepend the bos_token to each input.

    Returns
    -------
    Dict[str, Any]
        Tokenized data with input_ids and attention_mask.
    """
    # Add bos_token if needed
    text = []
    if add_bos_token:
        for t in element["text"]:
            t = tokenizer.bos_token + t
            text.append(t)
    else:
        text = element["text"]

    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=False,
        return_overflowing_tokens=False,
        padding=False,
        return_length=True,
    )

    outputs = {"text": [], "summary": [], "input_ids": [], "attention_mask": []}
    n_samples = len(inputs["input_ids"])
    for i in range(n_samples):
        input_ids = inputs.input_ids[i]
        attention_mask = inputs.attention_mask[i]
        length = inputs.length[i]

        # Length too long to process in a single sample, skip
        if length > max_length:
            continue

        # Left-pad for batch inference
        pad_tokens = max_length - length
        if pad_tokens:
            input_ids = [tokenizer.pad_token_id] * pad_tokens + input_ids
            attention_mask = [0] * pad_tokens + attention_mask

        # Update output
        outputs["text"].append(element["text"][i])
        outputs["summary"].append(element[summary_key][i])
        outputs["input_ids"].append(input_ids)
        outputs["attention_mask"].append(attention_mask)

    return outputs


def create_inference_dataset(
    *,
    tokenizer: Any,
    dataset_name: str,
    split: str,
    sample_size: Union[int, float],
    batch_size: int,
    max_length: int,
    random_state: int,
) -> DataLoader:
    """Create inference dataset.

    Parameters
    ----------
    tokenizer : Any
        Pretrained tokenizer.

    dataset_name : str
        Dataset name.

    split : str
        Dataset split.

    sample_size : Union[int, float]
        Sample size.

    batch_size : int
        Batch size.

    max_length : int
        Maximum length of sequence.

    random_state : int
        Random seed.

    Returns
    -------
    DataLoader
        Inference dataset.
    """
    if dataset_name == "xsum":
        raw = load_dataset(dataset_name, split=split)
        summary_key = "summary"
        columns = ["document", "id"]
    elif dataset_name == "samsum":
        raw = load_dataset(dataset_name, split=split)
        summary_key = "summary"
        columns = ["dialogue", "id"]
    else:
        raw = load_dataset(dataset_name, "3.0.0", split=split)
        summary_key = "highlights"
        columns = ["article", "id"]

    # Check if bos_token is needed
    test = "hey how are you?"
    input_ids = tokenizer(test).input_ids
    add_bos_token = input_ids[0] != tokenizer.bos_token_id

    # Preprocess
    try:
        prompts = getattr(PromptsInference, dataset_name)
    except AttributeError:
        raise AttributeError(
            f"There is no prompts method for dataset ({dataset_name}), define a method in the PromptsInference class "
            "for this dataset"
        )

    dataset = raw.map(
        prompts,
        batched=True,
        batch_size=min(1_000, len(raw)),
        num_proc=1,
        drop_last_batch=False,
        load_from_cache_file=False,
    )
    dataset = dataset.remove_columns(columns)

    # Tokenize
    dataset = dataset.map(
        tokenize_fn_inference,
        batched=True,
        batch_size=min(1_000, len(dataset)),
        num_proc=1,
        fn_kwargs={
            "tokenizer": tokenizer,
            "summary_key": summary_key,
            "max_length": max_length,
            "add_bos_token": add_bos_token,
        },
        remove_columns=dataset.column_names,
        drop_last_batch=False,
        load_from_cache_file=False,
    )

    # Recalculate sample size if float
    if type(sample_size) is float:
        sample_size = int(sample_size * len(dataset))

    if sample_size < len(dataset):
        # Subsample data
        model = SentenceTransformer("all-mpnet-base-v2", device=f"cuda:{EMBEDDINGS_DEVICE}")
        embeddings = model.encode(
            dataset["text"],
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            output_value="sentence_embedding",
            normalize_embeddings=True,
        )

        # Cluster
        clusterer = KMeans(n_clusters=sample_size, verbose=2, n_init="auto", random_state=random_state).fit(embeddings)
        selected = []
        for label in np.unique(clusterer.labels_):
            mask = clusterer.labels_ == label
            group = embeddings[mask]
            centroid = clusterer.cluster_centers_[label].reshape(1, -1)
            idx = np.where(mask)[0]
            metrics = cosine_similarity(centroid, group)
            selected.append(idx[metrics.argmax()])

        # Select indices
        dataset = dataset.select(selected)
        del model
        free_memory()

    # Create dataloader
    return DataLoader(dataset.with_format("torch"), batch_size=batch_size, shuffle=False, pin_memory=True)
