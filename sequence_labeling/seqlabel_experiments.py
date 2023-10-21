import argparse
import logging
import os
from copy import copy
import datetime
import time
from typing import List, Tuple
import gc

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split, StratifiedKFold
from spacy.tokens import Doc

from classif_experim import classif_experiment_runner
from classif_experim.classif_utils import classif_scores
from data_tools.create_spacy_dataset import load_spacy_dataset_docbin
from data_tools.seqlab_data.auto_gold_labels import GOLD_LABEL_AUTHOR
from data_tools.seqlab_data.create_spacy_span_dataset import get_annoation_tuples_from_doc, get_doc_id, get_doc_class
from data_tools.spacy_utils import ON_DOC_EXTENSION
from sequence_labeling.lab_experim_v0 import TASK_LABELS
from sequence_labeling.seqlab_sklearn_wrapper_multitask import OppSequenceLabelerMultitask
from sequence_labeling.seqlab_sklearn_wrapper_singletask import OppSequenceLabelerSingleTask
from sequence_labeling.span_f1_metric import compute_score_pr
from classif_experim.classif_experiment_runner import setup_logging

global logger


def filter_doc_spans(docs: List[Doc], keep, verbose=False):
    '''
    Remove, from each doc's spans, either all gold spans, or all spans that are not gold.
    :param keep: 'gold' or 'no-gold'
    :return:
    '''
    filtered_docs = []
    if verbose: print(f'KEEP: ', keep)
    for doc in docs:
        spans = get_annoation_tuples_from_doc(doc)
        if verbose: print(set([s[3] for s in spans]))
        if keep == 'gold': spans = [s for s in spans if s[3] == GOLD_LABEL_AUTHOR]
        elif keep == 'no-gold': spans = [s for s in spans if s[3] != GOLD_LABEL_AUTHOR]
        else: raise ValueError(f'Unknown value for keep: {keep}')
        if verbose: print(set([s[3] for s in spans]))
        doc_bytes = doc.to_bytes()
        # Deserialize the bytes to create a new Doc object
        new_doc = spacy.tokens.Doc(doc.vocab).from_bytes(doc_bytes)
        new_doc._.set(ON_DOC_EXTENSION, spans)
        filtered_docs.append(new_doc)
    if verbose: print()
    return filtered_docs

def run_crossvalid_seqlab_transformers(lang, model_label, model_params, num_folds=5, single_task=False,
                                       rnd_seed=3154561, test=False, pause_after_fold=0, allspans=False):
    '''
    Run x-fold crossvalidation for a given model, and report the results.
    '''
    logger.info(f'RUNNING crossvalid. for model: {model_label}')
    docs = load_spacy_dataset_docbin(lang, allspans=allspans)
    if test: docs = docs[:test]
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    # make columns list that has values 'P', 'R', 'F1' and f'{X}-F1', f'{X}-P', f'{X}-R' for all X in TASK_LABELS
    columns = ['F1', 'P', 'R'] + [f'{X}-F1' for X in TASK_LABELS] + [f'{X}-P' for X in TASK_LABELS] + [f'{X}-R' for X in TASK_LABELS]
    results_df = pd.DataFrame(columns=columns)
    rseed = rnd_seed
    classes = [get_doc_class(doc) for doc in docs]
    for train_index, test_index in foldgen.split(docs, classes):
        logger.info(f'Starting Fold {fold_index+1}')
        model = build_seqlab_model(model_label, rseed, model_params, single_task=single_task)
        logger.info(f'model built')
        # split data using the indices (these are not numpy or pandas arrays, so we can't use them directly)
        docs_train, docs_test = [], []
        for i in train_index: docs_train.append(docs[i])
        for i in test_index: docs_test.append(docs[i])
        if allspans:
            docs_train = filter_doc_spans(docs_train, keep='no-gold')
            docs_test = filter_doc_spans(docs_test, keep='gold')
        # train model
        model.fit_(docs_train)
        # evaluate model
        spans_test = [get_annoation_tuples_from_doc(doc) for doc in docs_test]
        spans_pred = model.predict(docs_test)
        del model
        scores = calculate_spanF1(docs_test, spans_test, spans_pred)
        scores_bin = calculate_binary_spanF1(spans_test, spans_pred)
        scores.update(scores_bin)
        scores_df = pd.DataFrame({fname: [fval] for fname, fval in scores.items()})
        # log scores
        logger.info(f'Fold {fold_index+1} scores:')
        #logger.info("; ".join([f"{fname:4}: {fval:.3f}" for fname, fval in scores.items()]))
        # Log global F1, P, R first
        logger.info("; ".join([f"{metric}: {scores[metric]:.3f}" for metric in ['F1', 'P', 'R']]))
        for label in TASK_LABELS: # Then, log each label-specific triplet on its own line
            logger.info("; ".join([f"{label}-{metric}: {scores[f'{label}-{metric}']:.3f}" for metric in ['F1', 'P', 'R']]))
            logger.info("; ".join([f"{label}-{metric}-b: {scores[f'{label}-{metric}-b']:.3f}" for metric in ['F1', 'P', 'R']]))
        # formatted_values = [f"{col:10}: {scores[col].iloc[0]:.3f}" for col in scores.columns]
        results_df = pd.concat([results_df, scores_df], ignore_index=True)
        if pause_after_fold and fold_index < num_folds - 1:
            logger.info(f'Pausing for {pause_after_fold} minutes...')
            time.sleep(pause_after_fold * 60)
        rseed += 1; fold_index += 1
    logger.info('CROSSVALIDATION results:')
    for fname in scores.keys():
        logger.info(f'{fname:8}: ' + '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    logger.info('Per-fold scores:')
    # for each score function, log all the per-fold results
    for fname in scores.keys():
        logger.info(f'{fname:8}: [{", ".join(f"{val:.3f}" for val in results_df[fname])}]')
    #print(results_df)

def build_seqlab_model(model_label, rseed, model_params, single_task=False):
    if not single_task:
        return OppSequenceLabelerMultitask(hf_model_label=model_label, rnd_seed=rseed, **model_params)
    else:
        return OppSequenceLabelerSingleTask(hf_model_label=model_label, rnd_seed=rseed, **model_params)

def run_seqlab_experiments(lang, num_folds, rnd_seed, test=False, experim_label='label',
                            pause_after_fold=0, pause_after_model=0, max_seq_length=256,
                            frequency_weights=False, taks_importance=None, single_task=False, allspans=False):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    single_task_label = "single_task" if single_task else "multi_task"
    log_filename = f"seqlabel_experiments_{experim_label}_{timestamp}_{single_task_label}.log"
    setup_logging(log_filename)
    global logger
    logger = classif_experiment_runner.logger
    models = HF_MODEL_LIST_SEQLAB[lang]
    HPARAMS = HF_CORE_HPARAMS_SEQLAB_SINGLETASK if single_task else HF_CORE_HPARAMS_SEQLAB_MULTITASK
    params = copy(HPARAMS)
    params['lang'] = lang
    params['eval'] = None
    params['max_seq_length'] = max_seq_length
    if not single_task:
        params['loss_freq_weights'] = frequency_weights
        params['task_importance'] = taks_importance
    logger.info(f'RUNNING classif. experiments: lang={lang.upper()}, num_folds={num_folds}, '
                f'max_seq_len={max_seq_length}, eval={params["eval"]}, rnd_seed={rnd_seed}, test={test}')
    logger.info(f'... HPARAMS = { "; ".join(f"{param}: {val}" for param, val in HPARAMS.items())}')
    logger.info(f'... ALL SPANS: {allspans}')
    logger.info(f'... SINGLE TASK: {single_task}')
    if not single_task:
        logger.info(f'... frequency_weights={frequency_weights}')
        logger.info(f'... task_importance={taks_importance}')
    init_batch_size = params['batch_size']
    for model in models:
        try_batch_size = init_batch_size
        grad_accum_steps = 1
        while try_batch_size >= 1:
            try:
                params['batch_size'] = try_batch_size
                params['gradient_accumulation_steps'] = grad_accum_steps
                run_crossvalid_seqlab_transformers(lang=lang, model_label=model, model_params=params, num_folds=num_folds,
                                       rnd_seed=rnd_seed, test=test, pause_after_fold=pause_after_fold, single_task=single_task,
                                       allspans=allspans)
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logging.warning(
                        f"GPU out of memory using batch size {try_batch_size}. Halving batch size and doubling gradient accumulation steps.")
                    try_batch_size //= 2
                    grad_accum_steps *= 2
                else:
                    raise e
            if try_batch_size < 1:
                logging.error("Minimum batch size reached and still encountering memory errors. Exiting.")
                break
        if pause_after_model:
            logger.info(f'Pausing for {pause_after_model} minutes...')
            time.sleep(pause_after_model * 60)

def spans_to_spanF1_format(ref_docs, spans: List[Tuple[str, int, int, str]]):
    '''
    Convert a list of (label, start, end, author) tuples to the format used by the spanF1 scorer:
    map text_id: spans, where spans is a list of span lists, one per label; for each label spans
    are in the form [label, set of character indices]
    :param spans:
    :return:
    '''
    result = {}
    for doc, span_list in zip(ref_docs, spans):
        text_id = get_doc_id(doc)
        if text_id not in result: result[text_id] = []
        labels = sorted(list(set([s[0] for s in span_list])))
        f1spans = []
        for l in labels:
            # take all spans with label l, and sort them by start index
            span_ranges = sorted([s[1:3] for s in span_list if s[0] == l], key=lambda x: x[0])
            # map each range to a set of character indices, using doc for offsets
            for start, end in span_ranges:
                first_char_index = doc[start].idx
                last_char_index = doc[end - 1].idx + len(doc[end - 1])
                f1spans.append([l, set(range(first_char_index, last_char_index))])
        result[text_id] = f1spans
    return result

def calculate_binary_spanF1(spans_test: List[List[Tuple[str, int, int, str]]],
                        spans_predict: List[List[Tuple[str, int, int, str]]]):
    '''
    Calculate binary classification metrics for occurr-vs-no-occur of a given label in the text.
    Predictions of occurrence are derived from the spans predicted by the model.
     '''
    scoring_fns = classif_scores('span-binary')
    scores = {}
    for label in TASK_LABELS:
        # for each position int the spans_test and spans_predic
        # we have a list of spans, each span is a list of 4 elements: label, start, end, author
        # we need to extract binary 0-1 prediction for each position:
        # 0 it there is no span with label equal to the current label, 1 otherwise
        spans_test_bin = [1 if label in [s[0] for s in spans] else 0 for spans in spans_test]
        spans_predict_bin = [1 if label in [s[0] for s in spans] else 0 for spans in spans_predict]
        # Now we can calculate the binary metrics
        for metric, score_fn in scoring_fns.items():
            scores[f'{label}-{metric}-b'] = score_fn(spans_test_bin, spans_predict_bin)
    return scores

def calculate_spanF1(ref_docs, spans_test: List[List[Tuple[str, int, int, str]]],
                        spans_predict: List[List[Tuple[str, int, int, str]]], disable_logger=True):
    spans_test_f1 = spans_to_spanF1_format(ref_docs, spans_test)
    spans_predict_f1 = spans_to_spanF1_format(ref_docs, spans_predict)
    return compute_score_pr(spans_predict_f1, spans_test_f1, TASK_LABELS, disable_logger=disable_logger)

def demo_experiment(lang, test_size=0.2, rnd_seed=1443):
    docs = load_spacy_dataset_docbin(lang)
    span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
    docs_train, docs_test, spans_train, spans_test = \
        train_test_split(docs, span_labels, test_size=test_size, random_state=rnd_seed)
    seq_lab = OppSequenceLabelerMultitask(num_train_epochs=1, empty_label_ratio=0.1, hf_model_label='microsoft/deberta-v3-base',
                                          lang=lang, eval=None, rnd_seed=rnd_seed)
    seq_lab.fit(docs_train, spans_train)
    spans_predict = seq_lab.predict(docs_test)
    calculate_spanF1(docs_test, spans_test, spans_predict, disable_logger=False)

HF_MODEL_LIST_SEQLAB = {
    'en': [
           'bert-base-cased',
           'roberta-base',
           'microsoft/deberta-v3-base',
           ],
    'es': [
            'dccuchile/bert-base-spanish-wwm-cased',
            'bertin-project/bertin-roberta-base-spanish',
          ],
}

HF_CORE_HPARAMS_SEQLAB_MULTITASK = {
    'learning_rate': 2e-5,
    'num_train_epochs': 10,
    'warmup': 0.1,
    'weight_decay': 0.01,
    'batch_size': 16,
}

HF_CORE_HPARAMS_SEQLAB_SINGLETASK = {
    'learning_rate': 2e-5,
    'num_train_epochs': 5,
    'warmup': 0.1,
    'weight_decay': 0.01,
    'batch_size': 16,
}

TASK_WEIGHTS_IGC = {
    'O': 1.0, 'A': 1.0, 'F': 2.0, 'P': 2.0, 'V': 1.0, 'E': 1.0,
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    '''
    Entry point function to accept command line arguments
    '''
    parser = argparse.ArgumentParser(description="Run Sequence Labelling Experiments")
    # Required arguments
    parser.add_argument("lang", type=str, help="Language")
    parser.add_argument("num_folds", type=int, help="Number of folds", default=5)
    parser.add_argument("rnd_seed", type=int, help="Random seed", default=42)
    # Optional arguments
    parser.add_argument("--test", type=int, default=0, help="Number of train examples to use, for test, if 0 use all (no test)")
    parser.add_argument("--experim_label", type=str, default=None, help="Experiment label")
    parser.add_argument("--pause_after_fold", type=int, default=0, help="Pause duration after fold")
    parser.add_argument("--pause_after_model", type=int, default=0, help="Pause duration after model")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--frequency_weights", action="store_true", help="Frequency weights flag")
    parser.add_argument("--taks_importance", type=str, default=None, help="Task importance")
    parser.add_argument("--single_task", type=str2bool, default=False, help="Single task mode flag")
    parser.add_argument("--allspans", type=str2bool, default=False, help="All spans flag")
    parser.add_argument("--gpu", type=int, default=0, help="index of the gpu for computation")
    # Parse the arguments
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    rnd_seed = args.rnd_seed
    print('ARGS TEST', args.test)
    test = False if args.test == 0 else args.test
    # Call the function with parsed arguments
    run_seqlab_experiments(
        lang=args.lang,
        num_folds=args.num_folds,
        rnd_seed=rnd_seed,
        test=test,
        experim_label=args.experim_label,
        pause_after_fold=args.pause_after_fold,
        pause_after_model=args.pause_after_model,
        max_seq_length=args.max_seq_length,
        frequency_weights=args.frequency_weights,
        taks_importance=args.taks_importance,
        single_task=args.single_task,
        allspans=args.allspans
    )


def batch_run_all(pause_after_experim=5):
    global logger
    for allspans in [False, True]:
        for single_task in [False, True]:
            for lang in ['en']:
                try:
                    run_seqlab_experiments(lang=lang, num_folds=5, rnd_seed=109432, test=False, single_task=single_task,
                                           pause_after_fold=10, pause_after_model=10, max_seq_length=256,
                                           frequency_weights=False, taks_importance=None, allspans=allspans)
                    logger.info(f'PAUSING AFTER EXPERIMENT for {pause_after_experim} minutes...')
                    logger.info(f'...')
                    time.sleep(pause_after_experim * 60)
                except Exception:
                    logger.exception(f'An error occured for lang={lang}, allspans={allspans}, single_task={single_task}')
                    logger.info(f'PAUSING AFTER ERROR for {pause_after_experim} minutes...')
                    logger.info(f'...')
                    time.sleep(pause_after_experim * 60)
                finally:
                    gc.collect()


if __name__ == '__main__':
    main()
    #batch_run_all()
