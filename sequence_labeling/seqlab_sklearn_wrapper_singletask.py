import os
import random
from typing import List, Tuple, Dict, Union

import datasets
import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc
from torch import tensor as TT
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer

from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.create_spacy_dataset import load_spacy_dataset_docbin
from data_tools.seqlab_data.create_spacy_span_dataset import get_annoation_tuples_from_doc, get_doc_id
from sequence_labeling.lab_experim_v0 import TASK_LABELS, TASK_INDICES
from sequence_labeling.multi_task_model import MultiTaskModel, Task
from sequence_labeling.spanannot2hf import extract_spans, convert_to_hf_format, labels_from_predictions, \
    align_labels_with_tokens, extract_span_ranges


class OppSequenceLabelerSingleTask(SklearnTransformerBase):
    '''
    'Oppositional Sequence Labeler', wraps the data transformation functionality and the multitask HF model
    for sequence labeling into a sklearn-like interface.
    '''

    def __init__(self, empty_label_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self._empty_label_ratio = empty_label_ratio

    def fit(self, docs: List[Doc], span_labels: List[Tuple[str, int, int, str]]):
        '''
        :param docs: spacy Docs
        :param span_labels: list of lists of spans, span is a tuple of (label, start, end, text),
                where start and end are token indices, ie, doc[start:end] is the span text
        :return:
        '''
        self._init_tokenizer()
        self._construct_datasets_for_inference(docs, span_labels)
        self._init_temp_folder()
        self._do_training()
        # input txt formatting and tokenization
        # training

    def fit_(self, docs: List[Doc]):
        '''
        Helper to enable fitting without previously extracting spans a separate list.
        docs: spacy Docs, with annotated spans, in the format defined in 'create_spacy_span_dataset.py'
        :return:
        '''
        span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
        self.fit(docs, span_labels)

    def _model_save_path(self, label):
        return os.path.join(self._tmp_folder, f'model-{label}')

    def _init_model(self, label):
        return MultiTaskModel(self._hf_model_label, [self._get_narrative_task(label)])

    def _do_training(self):
        self._is_roberta = 'roberta' in self._hf_model_label.lower()
        self._init_train_args()
        print('TEMP FOLDER: ', self._tmp_folder)
        self.models = {label: None for label in TASK_LABELS}
        for label in TASK_LABELS:
            self.models[label] = self._init_model(label)
            self._train_single_task_model(label)
            self.models[label].save_model(self._model_save_path(label))
            del self.models[label]
            self.models[label] = None

    def _train_single_task_model(self, label):
        model = self.models[label]
        dataset = self._datasets[label]
        train_dataset = dataset['train']
        eval_dataset = dataset['eval'] if self._eval else None
        # for index in random.sample(range(len(train_dataset)), 3):
        #     print(f"Sample {index} of the training set: {train_dataset[index]}.")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        #todo add compute_metrics appropriate for sequence labeling
        trainer = Trainer(model=model, args=self._training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset if self._eval else None,
            tokenizer=self.tokenizer, data_collator=data_collator)
        train_result = trainer.train()
        if self.models[label] is not trainer.model: # just in case
            del self.models[label]
            self.models[label] = trainer.model
        del trainer
        torch.cuda.empty_cache()

    def _construct_datasets_for_inference(self, docs, spans):
        self._construct_train_eval_raw_datasets(docs, spans)
        #self._inspect_data(self._raw_train, TASK_LABELS, num_samples=5)
        self._hf_tokenize_task_dataset()

    def _construct_train_eval_raw_datasets(self, docs, spans):
        ''' Construct 'raw' datasets, but separately for train end eval. '''
        if self._eval:
            docs_train, docs_eval, spans_train, spans_eval = \
                train_test_split(docs, spans, test_size=self._eval, random_state=self._rnd_seed)
            self._raw_train = self._construct_raw_hf_dataset(docs_train, spans_train, downsample=self._empty_label_ratio)
            self._raw_eval = self._construct_raw_hf_dataset(docs_eval, spans_eval, downsample=self._empty_label_ratio)
        else:
            self._raw_train = self._construct_raw_hf_dataset(docs, spans, downsample=self._empty_label_ratio)
            self._raw_eval = None

    def _construct_raw_hf_dataset(self, docs, span_labels, downsample) -> Dict[str, Dataset]:
        '''
        Convert the data to HF format: create one HF dataset per label, in BIO format.
        '''
        data_by_label = extract_spans(docs, span_labels, downsample_empty=downsample, rnd_seed=self._rnd_seed)
        datasets = {}
        for label, data in data_by_label.items():
            datasets[label] = convert_to_hf_format(label, data)
        return datasets

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_label)
        if isinstance(self.tokenizer, transformers.RobertaTokenizerFast):
            self.tokenizer.add_prefix_space = True
        self.tokenizer_params = {'truncation': True}
        if self._max_seq_length is not None: self.tokenizer_params['max_length'] = self._max_seq_length

    @classmethod
    def _get_narrative_task(self, task_label):
        '''
        Definition of a single task sequence labeling problem, compatible with MultiTaskModel.
        '''
        return Task(id=TASK_INDICES[task_label], name=None, num_labels=3, type="token_classification")

    def _calculate_task_weights(self):
        ''' Calculate task weights from task frequencies and importance weights. '''
        def normalize_weight_map(weights):
            sum_weights = sum(weights.values())
            for label in TASK_LABELS: weights[TASK_INDICES[label]] /= sum_weights
        if not self._loss_freq_weights and not self._task_importance:
            self._task_weights = None
        else: # calculate freq. weights, importance weights, or their combination if both are provided
            self._task_weights = {}
            if self._loss_freq_weights: # use loss frequency weights
                for label in TASK_LABELS: self._task_weights[TASK_INDICES[label]] = 1/self._task_frequencies[label]
                normalize_weight_map(self._task_weights)
                if self._task_importance:
                    for label in TASK_LABELS: self._task_weights[TASK_INDICES[label]] *= self._task_importance[label]
                    normalize_weight_map(self._task_weights)
            else:
                for label in TASK_LABELS: self._task_weights[TASK_INDICES[label]] = self._task_importance[label]
                normalize_weight_map(self._task_weights)

    def _hf_tokenize_task_dataset(self):
        '''
        Given a HF dataset for a single task (label) produced by _construct_hf_datasets,
        tokenize it, add taks labels, and return the tokenized dataset.
        '''
        # for each label, perform hf tokenization
        raw_dset_per_label = {}
        for label in TASK_LABELS:
            if not self._eval: raw_dataset = DatasetDict({'train': self._raw_train[label]})
            else: raw_dataset = DatasetDict({'train': self._raw_train[label], 'eval': self._raw_eval[label]})
            #label_list = raw_dataset['train'].features['ner_tags'].feature.names
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dataset, tokenizer=self.tokenizer, task_id=TASK_INDICES[label])
            raw_dset_per_label[label] = tokenized_dataset
        # shuffle datasets in raw_dset_per_label
        for label in TASK_LABELS:
            raw_dset_per_label[label].shuffle(seed=self._rnd_seed)
        self._datasets = raw_dset_per_label

    def _tokenize_token_classification_dataset(self, raw_datasets: Union[DatasetDict, Dataset], tokenizer, task_id,
                                               text_column_name='tokens', label_column_name='ner_tags'):
        # TODO extract as a class-level method
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=True,
                truncation=True,
                max_length=self._max_seq_length,
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None: label_ids.append(-100) # for special token set label to -100 (to ignore in loss)
                    elif word_idx != previous_word_idx: # set the label only for the first token of a word
                        label_ids.append(label[word_idx])
                    else: label_ids.append(-100) # for consecutive tokens of multi-token words, set the label to -100
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
            return tokenized_inputs
        if isinstance(raw_datasets, DatasetDict):
            tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
        elif isinstance(raw_datasets, Dataset): # helper code that enables to tokenize a single Dataset as DatasetDict
            dset = DatasetDict({'dset': raw_datasets})
            tokenized_datasets = dset.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
            tokenized_datasets = tokenized_datasets['dset']
        else: raise ValueError(f'Unknown dataset type: {type(raw_datasets)}')
        return tokenized_datasets

    def _inspect_data(self, datasets, label_list, num_samples=10):
        def print_aligned_tokens_and_tags(tokens, ner_tags):
            token_str = ' '.join([f"{token:<{len(token) + 2}}" for token in tokens])
            print(token_str)
            ner_tag_str = ' '.join([f"{str(tag):<{len(tokens[i]) + 2}}" for i, tag in enumerate(ner_tags)])
            print(ner_tag_str)
            print("\n" + "-" * 40)

        for label in label_list:
            dataset = datasets[label]
            sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            print(f"===== ORIGINAL TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = dataset[idx]['tokens']
                ner_tags = dataset[idx]['ner_tags']
                print_aligned_tokens_and_tags(tokens, ner_tags)
            tokenized_dataset = \
            self._tokenize_token_classification_dataset(DatasetDict({'train': dataset}),
                                                        self.tokenizer, TASK_INDICES[label])['train']
            print(f"\n===== TOKENIZED TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_dataset[idx]['input_ids'])
                labels = tokenized_dataset[idx]['labels']
                print_aligned_tokens_and_tags(tokens, labels)
            print("\n\n")

    def _construct_predict_dataset(self, docs, spans=None):
        if spans == None: # no spans provided, create a list of #docs empty lists for compatibility
            spans = [[] for _ in range(len(docs))]
        raw_dset_per_label = self._construct_raw_hf_dataset(docs, spans, downsample=None)
        tokenized_dset_per_label = {}
        for label in TASK_LABELS:
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dset_per_label[label], tokenizer=self.tokenizer, task_id=TASK_INDICES[label])
            tokenized_dset_per_label[label] = tokenized_dataset
        return tokenized_dset_per_label

    def predict(self, X: List[Doc]) -> List[List[Tuple[str, int, int, str]]]:
        '''
        :return: list of lists of spans (full annotations for one document); each span is a tuple of (label, start, end, author),
            for the data to be in the same format as in the original spacy data
        '''
        text_labels_pred = {} # intermediate map with output, and the helper function for adding labels to it
        def add_labels_to_map(label_map, text_id, task_label, labels: List[str]):
            if text_id not in label_map: label_map[text_id] = {}
            if task_label not in label_map[text_id]: label_map[text_id][task_label] = labels
            else: raise ValueError(f'Label map already contains labels for text id {text_id} and task {task_label}')
        # tokenize input, predict, transform data
        tokenized_dset_per_label = self._construct_predict_dataset(X)
        for label in TASK_LABELS:
            dset = tokenized_dset_per_label[label]
            model = MultiTaskModel.load_model(self._model_save_path(label), [self._get_narrative_task(label)])
            model.to(self.device)
            for t in dset:
                if not self._is_roberta: ttids = t['token_type_ids']
                else: ttids = [0]
                ids, att, tti, tsk = [t['input_ids']], [t['attention_mask']], [ttids], [TASK_INDICES[label]]
                ids, att, tti, tsk = TT(ids, device=self.device), TT(att, device=self.device), \
                                     TT(tti, device=self.device), TT(tsk, device=self.device)
                res, _ = model(ids, att, tti, task_ids=tsk)
                preds = res[0].cpu().detach().numpy()
                orig_tokens = t['tokens']
                pred_labels = labels_from_predictions(preds, t['labels'], label)
                pred_labels = align_labels_with_tokens(orig_tokens, pred_labels)
                add_labels_to_map(text_labels_pred, t['text_ids'], label, pred_labels)
            del model
            torch.cuda.empty_cache()
        # convert the map to the format of the original spacy data
        id2doc = {get_doc_id(doc): doc for doc in X}
        span_labels_pred = {text_id:[] for text_id in text_labels_pred.keys()}
        for text_id in text_labels_pred.keys():
            for task_label in text_labels_pred[text_id].keys():
                span_bio_tags = text_labels_pred[text_id][task_label]
                if len(span_bio_tags) == 0: continue
                doc = id2doc[text_id]
                # assert that doc has the same number of tokens as there are bio tags
                assert len(doc) == len(span_bio_tags)
                tokens = [token.text for token in doc]
                spans = extract_span_ranges(tokens, span_bio_tags, allow_hanging_itag=True)
                span_labels_pred[text_id].extend([(task_label, start, end, self._hf_model_label) for start, end in spans])
        return [span_labels_pred[get_doc_id(doc)] for doc in X]

def test_sklearn_wrapper(lang='en'):
    docs = load_spacy_dataset_docbin(lang)
    seq_lab = OppSequenceLabelerSingleTask(num_train_epochs=0.01, empty_label_ratio=0.1, hf_model_label='bert-base-cased', lang=lang, eval=None)
    seq_lab.fit_(docs)
    res = seq_lab.predict(docs[:10])
    print(res)

if __name__ == '__main__':
    test_sklearn_wrapper()