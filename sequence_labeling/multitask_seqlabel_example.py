'''

Code from:
https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240

'''
import logging
import os
import sys
import random
from typing import List

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from logging import getLogger
from dataclasses import dataclass

from torch import nn
from transformers import AutoModel, EvalPrediction, AutoTokenizer, DataCollatorForTokenClassification, Trainer, \
    default_data_collator, DataCollatorWithPadding, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed

from sequence_labeling.seqlabel_utils import DataTrainingArguments, ModelArguments

logger = getLogger()

def tokenize_token_classification_dataset(
    raw_datasets,
    tokenizer,
    task_id,
    label_list,
    text_column_name,
    label_column_name,
    data_args,
    training_args,
):

    label_to_id = {i: i for i in range(len(label_list))}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens"]

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
        )

    return tokenized_datasets


def load_token_classification_dataset(task_id, tokenizer, data_args, training_args):

    dataset_name = "conll2003"
    raw_datasets = load_dataset(dataset_name)

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = raw_datasets["train"].features[label_column_name].feature.names
    num_labels = len(label_list)
    print(task_id, label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name=dataset_name,
        num_labels=num_labels,
        type="token_classification",
    )

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )

@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int

def tokenize_seq_classification_dataset(
    tokenizer, raw_datasets, task_id, data_args, training_args
):
    sentence1_key, sentence2_key = "sentence1", "sentence2"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_text(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        examples["labels"] = examples.pop("label")
        result["task_ids"] = [task_id] * len(examples["labels"])
        return result

    def tokenize_and_pad_text(examples):
        result = tokenize_text(examples)

        examples["labels"] = [
            [l] + [-100] * (max_seq_length - 1) for l in examples["labels"]
        ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["idx", sentence1_key, sentence2_key]
        train_dataset = raw_datasets["train"].map(
            tokenize_and_pad_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )
        validation_dataset = raw_datasets["validation"].map(
            tokenize_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )

    return train_dataset, validation_dataset


def load_seq_classification_dataset(task_id, tokenizer, data_args, training_args):

    task_name = "mrpc"
    raw_datasets = load_dataset("glue", task_name, cache_dir=data_args.data_cache_dir)

    num_labels = len(raw_datasets["train"].features["label"].names)

    train_dataset, validation_dataset = tokenize_seq_classification_dataset(
        tokenizer,
        raw_datasets,
        task_id,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id, name=task_name, num_labels=num_labels, type="seq_classification"
    )

    return train_dataset, validation_dataset, task_info

def load_datasets(tokenizer, data_args, training_args):
    (
        seq_classification_train_dataset,
        seq_classification_validation_dataset,
        seq_classification_task,
    ) = load_seq_classification_dataset(0, tokenizer, data_args, training_args)
    (
        token_classification_train_dataset,
        token_classification_validation_dataset,
        token_classification_task,
    ) = load_token_classification_dataset(1, tokenizer, data_args, training_args)

    # Merge train datasets
    train_dataset_df = seq_classification_train_dataset.to_pandas().append(
        token_classification_train_dataset.to_pandas()
    )
    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    train_dataset.shuffle(seed=123)

    # Append validation datasets
    validation_dataset = [
        seq_classification_validation_dataset,
        token_classification_validation_dataset,
    ]

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    tasks = [seq_classification_task, token_classification_task]
    return tasks, dataset

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "token_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_ids=None,
            **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if preds.ndim == 2:
        # Token classification
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    elif preds.ndim == 3:
        # Sequence classification
        metric = load_metric("seqeval")

        predictions = np.argmax(preds, axis=2)

        true_predictions = [
            [f"tag-idx-{p}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]
        true_labels = [
            [f"tag-idx-{l}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]

        # Remove ignored index (special tokens)
        results = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        raise NotImplementedError()


def main(model_args, data_args, training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args)

    model = MultiTaskModel(model_args.encoder_name_or_path, tasks)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(data_args.max_eval_samples)))

            eval_datasets = new_ds

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "token_classification":
                data_collator = DataCollatorForTokenClassification(
                    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
                )
            else:
                if data_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                        tokenizer, pad_to_multiple_of=8
                    )
                else:
                    data_collator = None

            trainer.data_collator = data_collator
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_datasets)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    #model_args = ModelArguments(encoder_name_or_path="bert-base-cased")
    model_args = ModelArguments(encoder_name_or_path="bert-base-uncased")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="model-en-v3-spanbert",
        learning_rate=2e-5,
        num_train_epochs=1,
        #num_train_epochs=3,
        overwrite_output_dir=True,
    )
    data_args = DataTrainingArguments(task_name='mnli', max_seq_length=128)
    main(model_args, data_args, training_args)