'''

Adaptation of the code from:
https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240

'''
import logging
import os
import sys
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from datasets import load_metric, DatasetDict
from logging import getLogger

from torch import tensor as TT
from transformers import EvalPrediction, AutoTokenizer, DataCollatorForTokenClassification, Trainer, \
    default_data_collator, DataCollatorWithPadding, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed

from data_tools.seqlab_data.seqlabel_data_old import SHORT_LABELS_V2, LABELS_FIELD_NAME, TOKENS_FIELD_NAME, load_as_hf_dataset, \
    ORIG_GOLD_FIELD_NAME
from sequence_labeling.multi_task_model import MultiTaskModel, Task
from sequence_labeling.seqlabel_utils import DataTrainingArguments, ModelArguments

from settings import *

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
        #col_to_remove = ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens", ]
        col_to_remove = [LABELS_FIELD_NAME]

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
        )

    return tokenized_datasets

def load_narrative_token_classification_dataset(task_id, tokenizer, data_args, training_args):
    lang = data_args.language
    if lang == 'en': dfolder = SEQLABEL_ANNOTS_EN_JSON
    else: dfolder = SEQLABEL_ANNOTS_ES_JSON
    dataset_name = dfolder
    dset_train = load_as_hf_dataset(lang, task_id, 'train')
    dset_test = load_as_hf_dataset(lang, task_id, 'test')
    raw_datasets = DatasetDict({'train':dset_train['train'], 'test':dset_test['test']})
    text_column_name, label_column_name = TOKENS_FIELD_NAME, LABELS_FIELD_NAME
    label_list = raw_datasets['train'].features[label_column_name].feature.names
    num_labels = len(label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        TASK_INDICES[task_id],
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=TASK_INDICES[task_id],
        name=dataset_name,
        num_labels=num_labels,
        type="token_classification",
    )

    return (
        tokenized_datasets['train'],
        tokenized_datasets['test'],
        task_info,
    )

TASK_LABELS = sorted(list(SHORT_LABELS_V2.values()))
TASK_INDICES = { l:i for i, l in enumerate(TASK_LABELS) }

def get_narrative_tasks(lang):
    return [
        Task(
            id=TASK_INDICES[tl],
            name=f'{lang}-{tl}',
            num_labels=3,
            type="token_classification",
        )
        for tl in TASK_LABELS
    ]
def construct_narrative_tasks_datasets(tokenizer, data_args, training_args):
    per_task_dsets = [
        load_narrative_token_classification_dataset(t, tokenizer, data_args, training_args)
        for t in TASK_LABELS
    ]

    train_dataset_df = None
    for ds in per_task_dsets:
        train = ds[0]
        if train_dataset_df is None: train_dataset_df = train.to_pandas()
        else:
            train_dataset_df = train_dataset_df.append(train.to_pandas())
    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    train_dataset.shuffle(seed=1235)

    # Append validation datasets
    validation_dataset = [ ds[1] for ds in per_task_dsets ]

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    tasks = [ds[2] for ds in per_task_dsets]
    return tasks, dataset


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

def calc_avg_res(data):
    N = len(data); avg = sum(data)/N
    print(f'{avg:.3f}')

def setup_logger(training_args):
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

def load_checkpoint(training_args):
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
    return last_checkpoint

def run_train_eval(model_args, data_args, training_args):
    setup_logger(training_args)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = load_checkpoint(training_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tasks, raw_datasets = construct_narrative_tasks_datasets(tokenizer, data_args, training_args)

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
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

def run_train_eval_experiment(language='en'):
    model_args = model_arguments(language)
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="model-en-v4",
        learning_rate=2e-5,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        #per_device_gradient_accumulation_steps=1,
        overwrite_output_dir=True,
        resume_from_checkpoint=False,
    )
    data_args = DataTrainingArguments(task_name='mnli', max_seq_length=256, language=language)
    run_train_eval(model_args, data_args, training_args)


def model_arguments(language):
    if language == 'en':
        model = 'bert-base-uncased'
        #model = 'SpanBERT/spanbert-base-cased'
    elif language == 'es':
        model = 'dccuchile/bert-base-spanish-wwm-uncased'
    model_args = ModelArguments(
        encoder_name_or_path=model,
        use_fast_tokenizer=True,
    )
    return model_args

def labels_from_predictions(preds, tok_labels, task_id):
    tsk_label = TASK_LABELS[task_id]
    tags = {0: 'O', 1: f'B-{tsk_label}', 2: f'I-{tsk_label}'} # todo load from dataset data
    predictions = np.argmax(preds, axis=1)
    labels = [tags[p] for (p, l) in zip(predictions, tok_labels) if l != -100]
    return labels


def print_labeled_words(words, labels):
    labels = align_labels_with_words(words, labels)
    N = len(words)
    lens = [max(len(l), len(w)) for l, w in zip(labels, words)]
    print(' '.join(f'{words[i]:{lens[i]}}' for i in range(N)))
    print(' '.join(f'{labels[i]:{lens[i]}}' for i in range(N)))
    print()

def align_labels_with_words(words, labels):
    N = len(words)
    assert len(labels) <= N
    labels = labels + ['O'] * (N - len(labels))
    return labels

def output_propaganda_format(text_labels, text_words, fname, filter_ids=None):
    '''
    :param text_labels: mapping text_id -> map { task_id : labels }
    :param text_words: mapping text_id -> list of words (tokenized)
    :return:
    '''
    # calc. word index -> first char of word index for each text
    word_chars = {}
    for tid in text_words:
        words = text_words[tid]
        char_offsets = []; c = 0
        for i, w in enumerate(words):
            char_offsets.append(c)
            c += len(w)
        word_chars[tid] = char_offsets
    file_lines = []
    written_ids = set()
    for tid in text_words:
        if filter_ids and tid not in filter_ids: continue
        words = text_words[tid]
        print(f'TEXT ID: {tid}')
        for tsk_id, tsk_lab in enumerate(TASK_LABELS):
            labset = text_labels[tid][tsk_id] if tsk_id in text_labels[tid] else []
            used_labels = set()
            lab2str = lambda lbls: ';'.join(l for l in lbls)
            for labels in labset:
                if lab2str(labels) in used_labels: continue # possible duplication, due to dset structure
                else: used_labels.add(lab2str(labels))
                state = 'O'; start = 0
                fragments = []
                for i, l in enumerate(labels):
                    ltyp = l[0]
                    assert ltyp in ['O', 'B', 'I']
                    if state == 'O':
                        if ltyp == 'B': # ignore I if not preceeded by B
                            start = i
                            state = 'B'
                    elif state == 'B':
                        if ltyp == 'B' or ltyp == 'O':
                            fragments.append((start, i-1))
                            start = i
                            state = ltyp
                        elif ltyp == 'I': state = ltyp
                    elif state == 'I':
                        if ltyp == 'B' or ltyp == 'O':
                            fragments.append((start, i-1))
                            start = i
                            state = ltyp
                        elif ltyp == 'I': pass
                if state == 'B' or state == 'I': # end of text finished a fragment
                    fragments.append((start, len(labels)-1))
                print(tsk_lab)
                print_labeled_words(words, labels)
                char_offsets = word_chars[tid]
                for start, stop in fragments:
                    fw = words[start:stop+1]
                    print(f'[{";".join(fw)}]')
                    written_ids.add(tid)
                    file_lines.append({
                        'id': tid,
                        'categ': tsk_lab,
                        'start': char_offsets[start],
                        'end': char_offsets[stop]+len(words[stop])
                    })
    with open(fname, "w") as outf:
        for l in file_lines:
            fl = f"{l['id']}\t{l['categ']}\t{l['start']}\t{l['end']}"
            print(fl, file=outf)
    return written_ids

def evaluate_model(mfolder, lang, dev='cuda:0'):
    model_args = model_arguments(lang)
    tokenizer = AutoTokenizer.from_pretrained(
        mfolder,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print('loaded tokenizer')

    tasks = get_narrative_tasks(lang)
    model = MultiTaskModel(model_args.encoder_name_or_path, tasks)
    model.load_state_dict(torch.load(Path(mfolder)/'pytorch_model.bin'))
    print('loaded model')

    training_args = TrainingArguments(do_train=True, do_eval=True, output_dir="model-en-v3-spanbert",
                                      learning_rate=2e-5, num_train_epochs=10, overwrite_output_dir=True,
                                      resume_from_checkpoint=False)
    data_args = DataTrainingArguments(task_name='mnli', max_seq_length=256, language=lang)
    _, raw_datasets = construct_narrative_tasks_datasets(tokenizer, data_args, training_args)
    eval_datasets = raw_datasets["validation"]

    model.to(dev)
    model.eval()
    text_labels_pred = {}
    text_labels_gold = {}
    text_words = {}

    def add_labels_to_map(label_map, text_id, task_id, labels):
        if text_id not in label_map: label_map[text_id] = {}
        if task_id not in label_map[text_id]: label_map[text_id][task_id] = []
        label_map[text_id][task_id].append(labels)

    for eval_dataset, task in zip(eval_datasets, tasks):
        print(task.id)
        for t in eval_dataset:
            ids, att, tti, tsk = [t['input_ids']], [t['attention_mask']], [t['token_type_ids']], [task.id]
            ids, att, tti, tsk = TT(ids, device=dev), TT(att, device=dev), TT(tti, device=dev), TT(tsk, device=dev)
            res, _ = model(ids, att, tti, task_ids=tsk)
            preds = res[0].cpu().detach().numpy()
            text_id, words = t['text_id'], t['words']

            pred_labels = labels_from_predictions(preds, t['labels'], task.id)
            pred_labels = align_labels_with_words(words, pred_labels)
            gold_labels = t[ORIG_GOLD_FIELD_NAME].split(';')
            assert len(gold_labels) == len(pred_labels)

            if text_id not in text_words: text_words[text_id] = words
            else: assert words == text_words[text_id]

            add_labels_to_map(text_labels_pred, text_id, task.id, pred_labels)
            add_labels_to_map(text_labels_gold, text_id, task.id, gold_labels)

            # store words
        #print()
    assert set(text_labels_gold.keys()) == set(text_labels_pred.keys())
    gold_ids = output_propaganda_format(text_labels_gold, text_words, 'prop_gold.txt')
    output_propaganda_format(text_labels_pred, text_words, 'prop_pred.txt', filter_ids=gold_ids)

def print_label_data():
    to_long = {short:long for long, short in SHORT_LABELS_V2.items()}
    sind = sorted(list(TASK_INDICES.items()), key=lambda x: x[1])
    for l, i in sind: print(f'Index: {i}, label: {l}, {to_long[l]}')

if __name__ == "__main__":
    #run_train_eval_experiment(language='es')
    #evaluate_model('model-bert-en-v4', 'es')
    #run_train_eval_experiment(language='en')
    evaluate_model('model-en-v4', 'en')
    #print_label_data()
    #calc_avg_res([0.5667, 0.3164, 0.4122, 0.2683, 0.566, 0.6494])
    #evaluate_model('model-en-v2', 'en')
    #evaluate_model('model-es-v2', 'es')