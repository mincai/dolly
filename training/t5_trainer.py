# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)

from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
)

logger = logging.getLogger(__name__)
END_KEY = "<|endofoutput|>"


class CustomerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        # check data only
        # print(f" ==========DataCollatorForSeq2Seq: {batch}==========")
        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:

    inputs = tokenizer(batch["input_text"], max_length=max_length, truncation=True)
    labels = tokenizer(batch["output_text"], max_length=max_length, truncation=True)
    return {
        "input_ids": inputs.input_ids,
        "labels": labels.input_ids,
    }


def load_training_dataset(training_data_id: str = "", split: str = "train",
                          local_data_file_path: str = "") -> Dataset:
    if local_data_file_path: 
        logger.info(f"===============Loading local dataset from file: {local_data_file_path}=====================")
        dataset: Dataset = load_from_disk(local_data_file_path)[split]
    else:
        logger.info(f"===============Loading public dataset: {training_data_id}=====================")
        dataset: Dataset = load_dataset(training_data_id)[split]
    logger.info("Found %d rows", dataset.num_rows)

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> T5ForConditionalGeneration:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[T5ForConditionalGeneration, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, 
                       local_data_file_path: str = "") -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(local_data_file_path=local_data_file_path)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["input_text", "output_text"],
    )

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset

def train(
    local_output_dir,
    dbfs_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    deepspeed,
    gradient_checkpointing,
    local_rank,
    bf16,
    local_data_file_path="",
    test_size=1000,
    max_seq_length: int = None,
):
    set_seed(seed)
    gradient_checkpointing=False

    model, tokenizer = get_model_tokenizer(gradient_checkpointing=gradient_checkpointing)

    # Use the same max length that the model supports.  Try a couple different keys in case a different
    # model is used.  The default model uses n_positions.  If no config settings can be found just default
    # to 1024 as this is probably supported by most models.
    conf = model.config
    default_length = 1024
    if not max_seq_length:
        max_length: int = getattr(conf, "n_positions", getattr(conf, "seq_lenth", default_length))
        print(f"===========Use model config max_length : {max_length}====================")
          
    else:
        max_length: int = max_seq_length 
        print(f"===========Use customer config max_length : {max_length}====================")

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed,
                                           local_data_file_path=local_data_file_path)
    
    print(f" ============= processed_dataset: {processed_dataset[0]}=============")

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    data_collator = CustomerDataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors="pt", pad_to_multiple_of=8
    )

    if not dbfs_output_dir:
        logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
    )

    logger.info("Instantiating Trainer")
    model.config.decoder_start_token_id=0
    print(f"=======model.config.decoder_start_token_id: {model.config.decoder_start_token_id}=======")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option("--local-data-file-path", type=str, default="", help="""The local training data with list of json with `prompt` and `completion` as the key""")
@click.option("--test-size", type=int, default=1000, help="Test size.")
@click.option("--max-seq-length", type=int, default=None, help="Max sequence length.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
