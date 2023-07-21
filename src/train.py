"""Supervised fine-tuning LLM."""
import json
import logging
import math
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedPlugin, set_seed
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

from utils import PromptsTrain, free_memory

LOGGER = None
HERE = Path(__file__).resolve().parent
CONFIGS_DIR = HERE / "configs"
N_CPUS = cpu_count()

##########
# CONFIG #
##########


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for fine-tuning."""

    # Experiment config
    output_dir: str = field(default=None)
    seed: int = field(default=None)
    log_level: str = field(default="info")

    # Model config
    model_name_or_path: str = field(default=None)
    revision: str = field(default=None)

    # Training config
    deepspeed: str = field(default="configs/zero_stage2.json")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    learning_rate: float = field(default=2.0e-5)
    num_warmup_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    eval_every: int = field(default=100)
    save_strategy: str = field(default="steps")
    save_every: int = field(default=-1)
    save_best_checkpoint: bool = field(default=True)
    load_best_model: bool = field(default=True)

    # Data config
    dataset: str = field(default=None)
    overwrite_data: bool = field(default=False)
    max_length: int = field(default=1024)
    add_bos_token: bool = field(default=True)
    add_eos_token: bool = field(default=True)
    num_proc: int = field(default=1)
    max_train_samples: int = field(default=-1)
    max_validation_samples: int = field(default=-1)
    max_test_samples: int = field(default=-1)

    def __post_init__(self) -> None:
        """Post init stuff."""
        check_args_not_none(
            cls=self,
            args=[
                "output_dir",
                "seed",
                "model_name_or_path",
                "dataset",
            ],
        )

        # Create deepspeed plugin
        if self.deepspeed:
            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.deepspeed)
            self.deepspeed_plugin.gradient_accumulation_steps = self.gradient_accumulation_steps
            # Force to save whole unpartitioned model in output directory in ZeRO stage 3
            if self.deepspeed_plugin.zero_stage == 3:
                self.deepspeed_plugin.zero3_save_16bit_model = True

        if torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        if self.num_proc == -1:
            self.num_proc = N_CPUS

        # Turn off saving if <= 0
        if self.save_every <= 0:
            self.save_every = -1

        # Turn off eval if <= 0
        if self.eval_every <= 0:
            self.eval_every = -1

        # Force tensorboard logging
        self.report_to = ["tensorboard"]

        # Force directories inside output directory
        self.data_dir = str(Path(self.output_dir) / "data")
        self.cache_dir = str(Path(self.output_dir) / "cache")
        self.logging_dir = str(Path(self.output_dir) / "logs")
        self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")
        self.model_dir = str(Path(self.output_dir) / "model")

        # Create directories
        for arg in ["output_dir", "data_dir", "cache_dir", "checkpoint_dir", "model_dir"]:
            value = Path(getattr(self, arg))
            if not value.exists():
                value.mkdir(parents=True, exist_ok=True)

        # Copy configs
        shutil.copy(CONFIGS_DIR / "train.yaml", Path(self.output_dir))
        if self.deepspeed:
            shutil.copy(self.deepspeed, Path(self.output_dir) / "ds_config.json")


##########
# HELPER #
##########


def checkpoint_model(
    *,
    checkpoint_dir: str,
    checkpoint_id: str,
    model: Any,
    epoch: int,
    last_global_step: int,
    **kwargs: Any,
) -> None:
    """Save model at checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Checkpoint directory.

    checkpoint_id : str
        Checkpoint ID.

    model : Any
        Pretrained model to save.

    epoch : int
        Last completed training epoch.

    last_global_step : int
        Last completed training optimization step.

    **kwargs : Any
        Additional keyword arguments.
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    path = os.path.join(checkpoint_dir, checkpoint_id)
    if Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)

    success = model.save_checkpoint(checkpoint_dir, checkpoint_id, checkpoint_state_dict)
    status_msg = f"Checkpointing: checkpoint_dir={checkpoint_dir}, checkpoint_id={checkpoint_id}"
    if success:
        LOGGER.info(f"Success {status_msg}")
    else:
        LOGGER.warning(f"Failure {status_msg}")


def load_training_checkpoint(
    *, checkpoint_dir: str, model: Any, tag: Optional[str] = None, **kwargs: Any
) -> Tuple[int, int]:
    """Load model from checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Checkpoint directory.

    model : Any
        Pretrained model to load.

    tag : str, default=None
        Checkpoint tag.

    **kwargs : Any
        Additional keyword arguments for loading checkpoint.

    Returns
    -------
    Tuple[int, int]
        Epoch and last completed step.
    """
    _, checkpoint_state_dict = model.load_checkpoint(checkpoint_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict

    return epoch, last_global_step


def evaluate(*, model: Any, dataloader: DataLoader, accelerator: Accelerator) -> Tuple[float, float]:
    """Evaluate model on dataset.

    Parameters
    ----------
    model : Any
        Pretrained model.

    dataloader : DataLoader
        Dataloader.

    accelerator : Accelerator
        Accelerator object.

    Returns
    -------
    Tuple[float, float]
        Perplexity and loss metrics.
    """
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss))

    losses = torch.cat(losses)
    try:
        validation_loss = torch.mean(losses)
        perplexity = math.exp(validation_loss)
    except OverflowError:
        perplexity = float("inf")

    free_memory()

    return perplexity, validation_loss.item()


def check_args_not_none(*, cls: Any, args: List[str]) -> None:
    """Check for required arguments in dataclass.

    Parameters
    ----------
    cls : Any
        Dataclass.

    args : List[str]
        List of arguments that should be present.
    """
    for arg in args:
        if getattr(cls, arg) is None:
            raise ValueError(f"Missing argument ({arg}) for class {cls.__class__.__name__}")


def tokenize_fn(
    element: Dict[str, Any],
    *,
    tokenizer: Any,
    key: str,
    max_length: int,
    add_bos_token: bool,
    add_eos_token: bool,
) -> Dict[str, Any]:
    """Tokenize function for training (right padding).

    Parameters
    ----------
    element : Dict[str, Any]
        Input data.

    tokenizer : Any
       Pretrained tokenizer.

    key : str
        How to access text in element.

    max_length : int
        Maximum sequence length.

    add_bos_token : bool
        Whether to prepend the bos_token to each input.

    add_eos_token : bool
        Whether to append the eos_token to each input.

    Returns
    -------
    Dict[str, Any]
        Tokenized data with input_ids and attention_mask.
    """
    # Add special tokens if needed
    text = []
    if add_bos_token or add_eos_token:
        for t in element[key]:
            if add_bos_token:
                t = tokenizer.bos_token + t
            if add_eos_token:
                t = t + tokenizer.eos_token
            text.append(t)
    else:
        text = element[key]

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
        padding=False,
        return_length=True,
    )

    outputs = {"input_ids": [], "labels": [], "attention_mask": []}
    n_samples = len(inputs["input_ids"])
    for i in range(n_samples):
        input_ids = inputs.input_ids[i]
        length = inputs.length[i]

        # Truncation happened because input is too long, ignore sample
        if length == max_length and input_ids[-1] != tokenizer.eos_token_id and add_eos_token:
            continue

        attention_mask = [1] * len(input_ids)
        labels = deepcopy(input_ids)

        # Check if padding needed
        pad_tokens = max_length - len(input_ids)
        if pad_tokens:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_tokens
            attention_mask = attention_mask + [0] * pad_tokens
            labels = labels + [-100] * pad_tokens

        # Append samples
        outputs["input_ids"].append(input_ids)
        outputs["attention_mask"].append(attention_mask)
        outputs["labels"].append(labels)

    return outputs


def create_dataset(*, config: TrainingArguments, tokenizer: Any) -> None:
    """Creates dataset.

    Parameters
    ----------
    config : TrainingArguments
        Configuration.

    tokenizer : Any
        Pretrained tokenizer.
    """
    data_dir = Path(config.data_dir)

    # Clean up previous dataset if needed
    if data_dir.exists() and config.overwrite_data:
        shutil.rmtree(str(data_dir))
        data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "dataset_dict.json").exists():
        LOGGER.info("Building new dataset")
        raw = (
            load_dataset(config.dataset, "3.0.0", cache_dir=config.cache_dir)
            if config.dataset == "cnn_dailymail"
            else load_dataset(config.dataset, cache_dir=config.cache_dir)
        )

        try:
            prompts = getattr(PromptsTrain, config.dataset)
        except AttributeError:
            raise AttributeError(
                f"There is no prompts method for dataset ({config.dataset}), define a method in the PromptsTrain class "
                "for this dataset"
            )

        # Preprocess
        dataset = {}
        for split in raw.keys():
            LOGGER.info(f"Creating prompts for dataset split ({split})")
            ds = raw[split].map(
                prompts,
                batched=True,
                batch_size=min(1_000, len(raw)),
                num_proc=config.num_proc,
                load_from_cache_file=not config.overwrite_data,
                drop_last_batch=False,
            )
            columns = ["document", "summary", "id"] if config.dataset == "xsum" else ["article", "highlights", "id"]
            ds = ds.remove_columns(columns)

            # Tokenize
            LOGGER.info(f"Tokenizing dataset split ({split})")
            dataset[split] = ds.map(
                tokenize_fn,
                batched=True,
                batch_size=min(1_000, len(ds)),
                num_proc=config.num_proc,
                load_from_cache_file=not config.overwrite_data,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "key": "text",
                    "max_length": config.max_length,
                    "add_bos_token": config.add_bos_token,
                    "add_eos_token": config.add_eos_token,
                },
                remove_columns=ds.column_names,
                drop_last_batch=False,
            )

        # Save dataset
        LOGGER.info(f"Saving new dataset to disk in {config.output_dir + '/data'}")
        dataset = DatasetDict(dataset)
        dataset.save_to_disk(data_dir)


########
# MAIN #
########


def train(*, accelerator: Accelerator, config: TrainingArguments) -> None:
    """Main training function."""
    # Set the training seed
    set_seed(config.seed)

    #########
    # MODEL #
    #########

    LOGGER.info(f"Loading pretrained language model {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        cache_dir=config.cache_dir,
        use_cache=False if config.gradient_checkpointing else True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    #############
    # TOKENIZER #
    #############

    LOGGER.info(f"Loading tokenizer for {config.model_name_or_path} language model")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        use_fast=True,
        cache_dir=config.cache_dir,
    )

    # Add special tokens
    added_tokens = 0
    for key, value in {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}.items():
        attribute = getattr(tokenizer, key, None)
        if attribute is None:
            LOGGER.info(f"Token {key} is None, adding special token")
            added_tokens += tokenizer.add_special_tokens({key: value})
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # Check and see if the bos and eos tokens are added automatically to tokenized inputs
    test = "hey how are you?"
    input_ids = tokenizer(test).input_ids
    add_bos_token = input_ids[0] != tokenizer.bos_token_id
    add_eos_token = input_ids[-1] != tokenizer.eos_token_id

    # Config overrides
    if not config.add_bos_token:
        add_bos_token = False
    if not config.add_eos_token:
        add_eos_token = False

    config.add_bos_token = add_bos_token
    config.add_eos_token = add_eos_token

    ############
    # DATASETS #
    ############

    if accelerator.is_main_process:
        create_dataset(config=config, tokenizer=tokenizer)
    accelerator.wait_for_everyone()

    # Load dataset from disk
    with accelerator.main_process_first():
        LOGGER.info("Loading dataset from disk")
        dataset = load_from_disk(config.data_dir)

        # Subsample dataset
        if 0 < config.max_train_samples < len(dataset["train"]):
            LOGGER.info(f"Subsampling dataset split 'train' to use {config.max_train_samples} samples")
            idx = np.random.choice(len(dataset["train"]), size=config.max_train_samples)
            dataset["train"] = dataset["train"].select(idx)
        if 0 < config.max_validation_samples < len(dataset["test"]):
            LOGGER.info(f"Subsampling dataset split 'validation' to use {config.max_validation_samples} samples")
            idx = np.random.choice(len(dataset["validation"]), size=config.max_validation_samples)
            dataset["validation"] = dataset["validation"].select(idx)
        if 0 < config.max_test_samples < len(dataset["test"]):
            LOGGER.info(f"Subsampling dataset split 'test' to use {config.max_test_samples} samples")
            idx = np.random.choice(len(dataset["test"]), size=config.max_test_samples)
            dataset["test"] = dataset["test"].select(idx)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset["train"].with_format("torch"),
        collate_fn=DefaultDataCollator(),
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        dataset["validation"].with_format("torch"),
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset["test"].with_format("torch"),
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
    )

    #############
    # CONFIGURE #
    #############

    # Creates dummy optimizer and scheduler
    optimizer = DummyOptim(model.parameters(), lr=config.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = DummyScheduler(
        optimizer, total_num_steps=config.max_train_steps, warmup_num_steps=config.num_warmup_steps
    )

    # Prepare for training
    # Note: This will automatically push data batch to GPU during loop
    model, optimizer, lr_scheduler, train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    )

    #########
    # TRAIN #
    #########

    # Recalculate total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    total_batch_size = (
        config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    )
    LOGGER.info("***** Running training *****")
    LOGGER.info(f"Num examples = {len(train_dataloader.dataset)}")
    LOGGER.info(f"Num epochs = {config.num_train_epochs}")
    LOGGER.info(f"Instantaneous batch size per device = {config.per_device_train_batch_size}")
    LOGGER.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    LOGGER.info(f"Gradient accumulation steps = {config.gradient_accumulation_steps}")
    LOGGER.info(f"Total optimization steps = {config.max_train_steps}")

    # Only show the progress bar once on each machine
    pbar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    pbar_metrics = {
        "train_loss": float("inf"),
        "validation_loss": float("inf"),
        "perplexity": float("inf"),
    }
    pbar.set_postfix(pbar_metrics)

    # Begin training loop
    starting_epoch = 1
    all_metrics = []
    total_loss = 0.0
    total_steps = 0
    completed_steps = 0
    n_batches = len(train_dataloader)
    best_metric = float("inf")
    for epoch in range(starting_epoch, config.num_train_epochs + 1):
        for step, batch in enumerate(train_dataloader, 1):
            perc = round(100 * step / n_batches, 1)
            pbar.set_description(f"epoch {epoch} - batch: {step}/{n_batches} ({perc}%)")
            model.train()

            # Accumulate gradients
            if not (step % config.gradient_accumulation_steps == 0 or step == len(train_dataloader)):
                with accelerator.no_sync(model):
                    # Forward pass
                    outputs = model(**batch)

                    # Backwards pass
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss = loss / config.gradient_accumulation_steps
                    accelerator.backward(loss)
                    total_steps += 1

            # Sync gradients and run optimization step
            else:
                # Forward pass
                outputs = model(**batch)

                # Backwards pass
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss = loss / config.gradient_accumulation_steps
                accelerator.backward(loss)
                total_steps += 1

                # Optimization
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                pbar.update(1)
                pbar_metrics["train_loss"] = round(total_loss.item() / total_steps, 2)
                pbar.set_postfix(pbar_metrics)

            # Break if exceed max train steps
            if completed_steps >= config.max_train_steps:
                break

            # At least 1 optimization step completed
            if completed_steps > 0:
                # Checkpoint model
                if (
                    config.save_strategy == "steps"
                    and completed_steps % config.save_every == 0
                    and config.save_every > 0
                ):
                    checkpoint_id = f"step_{completed_steps}"
                    path = Path(config.checkpoint_dir) / checkpoint_id
                    LOGGER.info(
                        f"Saving model checkpoint at step {completed_steps} to {path}",
                        main_process_only=False,
                    )
                    checkpoint_model(
                        checkpoint_dir=config.checkpoint_dir,
                        checkpoint_id=checkpoint_id,
                        model=model,
                        epoch=epoch,
                        last_global_step=completed_steps,
                    )

                # Evaluate
                if (
                    config.eval_strategy == "steps"
                    and completed_steps % config.eval_every == 0
                    and config.eval_every > 0
                ):
                    LOGGER.info(f"Evaluating model at step {completed_steps}")
                    perplexity, validation_loss = evaluate(
                        model=model,
                        dataloader=validation_dataloader,
                        accelerator=accelerator,
                    )

                    metrics = {
                        "train_loss": total_loss.item() / total_steps,
                        "perplexity": perplexity,
                        "validation_loss": validation_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    all_metrics.append(metrics)

                    LOGGER.info(
                        f"step {completed_steps} - perplexity: {perplexity} validation_loss: {validation_loss}",
                    )
                    accelerator.log(metrics, step=completed_steps)

                    pbar_metrics["train_loss"] = round(metrics["train_loss"], 2)
                    pbar_metrics["validation_loss"] = round(metrics["validation_loss"], 2)
                    pbar_metrics["perplexity"] = round(metrics["perplexity"], 2)
                    pbar.set_postfix(pbar_metrics)

                    if perplexity < best_metric:
                        best_metric = perplexity
                        LOGGER.info(f"New best metric {best_metric} at step {completed_steps}")

                        if config.save_best_checkpoint:
                            # Checkpoint model
                            checkpoint_id = "best"
                            path = Path(config.checkpoint_dir) / checkpoint_id
                            LOGGER.info(
                                f"Saving best model checkpoint at step {completed_steps} to {path}",
                                main_process_only=False,
                            )
                            checkpoint_model(
                                checkpoint_dir=config.checkpoint_dir,
                                checkpoint_id=checkpoint_id,
                                model=model,
                                epoch=epoch,
                                last_global_step=completed_steps,
                            )

        # End of training epoch

        # Checkpoint model
        if config.save_strategy == "epoch" and epoch % config.save_every == 0 and config.save_every > 0:
            checkpoint_id = f"epoch_{epoch}"
            path = Path(config.checkpoint_dir) / checkpoint_id
            LOGGER.info(
                f"Saving model checkpoint at epoch {epoch} to {path}",
                main_process_only=False,
            )
            checkpoint_model(
                checkpoint_dir=config.checkpoint_dir,
                checkpoint_id=checkpoint_id,
                model=model,
                epoch=epoch,
                last_global_step=completed_steps,
            )

        # Evaluate
        if config.eval_strategy == "epoch" and epoch % config.eval_every == 0 and config.eval_every > 0:
            LOGGER.info(f"Evaluating model at epoch {epoch}")
            perplexity, validation_loss = evaluate(
                model=model,
                dataloader=validation_dataloader,
                accelerator=accelerator,
            )

            metrics = {
                "train_loss": total_loss.item() / total_steps,
                "perplexity": perplexity,
                "validation_loss": validation_loss,
                "epoch": epoch,
                "step": completed_steps,
            }
            all_metrics.append(metrics)

            LOGGER.info(
                f"epoch {epoch} - perplexity: {perplexity} validation_loss: {validation_loss}",
            )
            accelerator.log(metrics, step=completed_steps)

            pbar_metrics["train_loss"] = round(metrics["train_loss"], 2)
            pbar_metrics["validation_loss"] = round(metrics["validation_loss"], 2)
            pbar_metrics["perplexity"] = round(metrics["perplexity"], 2)
            pbar.set_postfix(pbar_metrics)

            if perplexity < best_metric:
                best_metric = perplexity
                LOGGER.info(f"New best metric {best_metric} at epoch {epoch}")

                if config.save_best_checkpoint:
                    # Checkpoint model
                    checkpoint_id = "best"
                    path = Path(config.checkpoint_dir) / checkpoint_id
                    LOGGER.info(
                        f"Saving best model checkpoint at epoch {epoch} to {path}",
                        main_process_only=False,
                    )
                    checkpoint_model(
                        checkpoint_dir=config.checkpoint_dir,
                        checkpoint_id=checkpoint_id,
                        model=model,
                        epoch=epoch,
                        last_global_step=completed_steps,
                    )

    # Final model evaluation
    if config.eval_every > 0:
        LOGGER.info("Training finished, evaluating model")
        perplexity, validation_loss = evaluate(
            model=model,
            dataloader=validation_dataloader,
            accelerator=accelerator,
        )

        metrics = {
            "train_loss": total_loss.item() / total_steps,
            "perplexity": perplexity,
            "validation_loss": validation_loss,
            "epoch": epoch,
            "step": completed_steps,
        }
        all_metrics.append(metrics)

        LOGGER.info(
            f"Training finished - perplexity: {perplexity} validation_loss: {validation_loss}",
        )
        accelerator.log(metrics, step=completed_steps)

        pbar_metrics["train_loss"] = round(metrics["train_loss"], 2)
        pbar_metrics["validation_loss"] = round(metrics["validation_loss"], 2)
        pbar_metrics["perplexity"] = round(metrics["perplexity"], 2)
        pbar.set_postfix(pbar_metrics)

        if perplexity < best_metric:
            best_metric = perplexity
            LOGGER.info(f"New best metric {best_metric} at training end")

            if config.save_best_checkpoint:
                # Checkpoint model
                checkpoint_id = "best"
                path = Path(config.checkpoint_dir) / checkpoint_id
                LOGGER.info(f"Saving best model checkpoint at training end to {path}", main_process_only=False)
                checkpoint_model(
                    checkpoint_dir=config.checkpoint_dir,
                    checkpoint_id=checkpoint_id,
                    model=model,
                    epoch=epoch,
                    last_global_step=completed_steps,
                )

    ########
    # SAVE #
    ########

    # Load best model
    if config.load_best_model:
        load_training_checkpoint(
            checkpoint_dir=config.checkpoint_dir,
            model=model,
            tag="best",
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
    validation_perplexity, validation_loss = evaluate(
        model=model,
        dataloader=validation_dataloader,
        accelerator=accelerator,
    )
    test_perplexity, test_loss = evaluate(
        model=model,
        dataloader=test_dataloader,
        accelerator=accelerator,
    )
    LOGGER.info(
        f"Final model metrics - validation_perplexity: {validation_perplexity}, validation_loss: {validation_loss} | "
        f"test_perplexity: {test_perplexity}, test_loss: {test_loss}"
    )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
    # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
    # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
    # For Zero Stages 1 and 2, models are saved as usual in the output directory.
    # The model name saved is `pytorch_model.bin`
    unwrapped_model.save_pretrained(
        config.model_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(config.model_dir)

    # Save final model metrics
    with open(os.path.join(config.model_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "validation_loss": validation_loss,
                "validation_perplexity": validation_perplexity,
                "test_loss": test_loss,
                "test_perplexity": test_perplexity,
            },
            f,
        )

    # Save all metrics if kept during training
    if all_metrics:
        with open(os.path.join(config.output_dir, "all_metrics.json"), "w") as f:
            json.dump(all_metrics, f)

    accelerator.end_training()


if __name__ == "__main__":
    # Parse CLI
    parser = transformers.HfArgumentParser(TrainingArguments)
    config = parser.parse_yaml_file(CONFIGS_DIR / "train.yaml")[0]

    # Copy full config into output_dir to help with debugging
    with open(Path(config.output_dir) / "train_full.yaml", "w") as outfile:
        yaml.dump(config.to_dict(), outfile, default_flow_style=False)

    # Initialize the accelerator
    accelerator = Accelerator(
        log_with=config.report_to,
        project_dir=config.output_dir,
        deepspeed_plugin=config.deepspeed_plugin,
    )
    accelerator.init_trackers(project_name="logs")

    # Make one log on every process with the configuration for debugging
    LOGGER = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=config.log_level.upper(),
    )

    # Update local ranks
    config.local_rank = int(os.environ["LOCAL_RANK"])
    assert config.local_rank != -1, "BAD THINGS ARE ABOUT TO HAPPEN!"
    LOGGER.info(f"Configuring local ranks: I am local process: {config.local_rank}", main_process_only=False)

    train(accelerator=accelerator, config=config)
