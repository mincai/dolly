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
import torch
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
)

from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
)

END_KEY = "<|endofsentence|>"
RESPONSE_KEY_NL = "\nagent:"
CHAT_START_KEY = "\n\n###\n\nagent"
logger = logging.getLogger(__name__)

def split_first_agent(instruction):
    pos = instruction.find(CHAT_START_KEY)
    if pos != -1:
        first_agent_reply = "\n" + instruction[pos+len(CHAT_START_KEY)-len(RESPONSE_KEY_NL):].replace("\n", "")
    else:
        first_agent_reply = ""
    return instruction[:pos], first_agent_reply


def process_chat(chat: str):
    end_key = END_KEY
    chats_list = chat.split(end_key)
    instruction, first_agent_reply = split_first_agent(chats_list[0])
    if first_agent_reply:
        chats_list = [instruction, first_agent_reply] + chats_list[1:]
    else:
        chats_list = [instruction] + chats_list[1:]
    return chats_list


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
        assert len(response_token_ids) == 1

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                print(f"========== {examples[i]}; {labels[i]}; {response_token_ids[0]}==========")
                response_token_ids_start_idx = -2
                #raise RuntimeError(f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}')

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(training_data_id: str = "", split: str = "train",
                          local_data_file_path:str="") -> Dataset:
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
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, RESPONSE_KEY_NL]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, 
        torch_dtype=torch.float16,
        use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
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
        remove_columns=["text", "ticket_uuid"],
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
    pretrained_model_name=DEFAULT_INPUT_MODEL,
    local_data_file_path="",
    test_size=1000,
    max_frozen_layers: int = 30,
    max_seq_length: int = None,
):
    set_seed(seed)
    gradient_checkpointing=False

    model, tokenizer = get_model_tokenizer(pretrained_model_name_or_path=pretrained_model_name,
                                           gradient_checkpointing=gradient_checkpointing)

    # Use the same max length that the model supports.  Try a couple different keys in case a different
    # model is used.  The default model uses n_positions.  If no config settings can be found just default
    # to 1024 as this is probably supported by most models.
    conf = model.config
    default_length = 1024
    if not max_seq_length:
        max_length: int = getattr(conf, "n_positions", getattr(conf, "seq_lenth", getattr(conf, "max_position_embeddings", default_length)))
        print(f"===========Use model config max_length {model.config} : {max_length}====================")
          
    else:
        max_length: int = max_seq_length 
        print(f"===========Use customer config max_length : {max_length}====================")
    

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed,
                                           local_data_file_path=local_data_file_path)

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    if not dbfs_output_dir:
        logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=True,
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
    
    frozen_layers_prefix = [f"gpt_neox.layers.{i}" for i in range(max_frozen_layers)] + ["embed_in"]
    print(f"========== Frozen_layers_prefix:\n {frozen_layers_prefix}")
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f" ======= Before frozen earlier layer, total trainable parameters: {params} =================== ")
    
    trainable_layers = []
    for name, param in model.named_parameters():
        trainable = True
        for layer_prefix in frozen_layers_prefix:
            if layer_prefix in name:
                param.requires_grad = False
                trainable = False
        if trainable:
            trainable_layers.append(name)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f" ========= After frozen earlier layer, trainable parameters: {params}! There are {trainable_layers} ")
    

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
@click.option("--pretrained-model-name", type=str, default=DEFAULT_INPUT_MODEL, help="""Model name""")
@click.option("--test-size", type=int, default=1000, help="Test size.")
@click.option("--max-seq-length", type=int, default=None, help="Max sequence length.")
@click.option("--max-frozen-layers", type=int, default=0, help="Max sequence length.")
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
