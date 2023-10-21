'''

Methods for loading and preparing the sequence labeling data.

'''

import json
import math
from copy import deepcopy
from itertools import groupby
from pathlib import Path, PurePath
import random
from typing import List

import datasets
import numpy as np

import warnings

import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from data_tools.seqlab_data.data_struct import AuthLabelTxtSpans, SpanAnnotation
from data_tools.seqlab_data.labstudio_parser import LabelStudioParser, ls_parse_all_subfolders
from data_tools.seqlab_data.seqlab_utils import match_span_annots_to_spacy_spans_old, group_by_perm
from data_tools.seqlab_data.url_tokenization import SpacyURLTokenizer
from settings import *

warnings.filterwarnings("ignore")

# util functions

def parse_lang_from_folder_name(folder):
    folder = PurePath(folder).name
    if '-en-' in folder: return 'en'
    elif '-es-' in folder: return 'es'
    else: raise ValueError(f'Cannot extract language from folder name: {folder}')

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

def group_annots_by_author_text(annots):
    keyf = lambda a: (a['file_id'], a['author'])
    annots = sorted(annots, key=keyf)
    gb = groupby(annots, keyf)
    res = { k: list(g) for k, g in gb }
    return res

def print_annot_stats(annots):
    by_txt = group_by_perm(annots, lambda a: a.text_id)
    num_txt = len(by_txt)
    print(f'Total number of annotated texts: {num_txt}')
    N = len(annots)
    print(f'Total number of annotations: {N}')
    print(f' Avg. number of annotations per text: {N / num_txt:.3f}')

PER_FILE_OUT_FOLDER = 'per-file-data'

TOKENS_FIELD_NAME = 'words'
LABELS_FIELD_NAME = 'narrative_tag'
TEXT_ID_FIELD_NAME = 'text_id'
ORIG_GOLD_FIELD_NAME = 'orig_gold'

class LS2HugginFaceLabelSepNew:
    '''
    Converts Label Studio annotations to the HuggingFace dataset format for sequence labeling.
    Creates a separate dataset for each of the label - in order to train a separate model for each label.
    '''

    def __init__(self, ls_parser: LabelStudioParser, lang, rseed=7120123, selectx=0.5,
                    test_size=0.25, test_only=False, split_method='stratify', verbose=False):
        set_random_seed(rseed)
        self._rseed = rseed
        self._ls_parser = ls_parser
        self._lang = lang
        self._verbose = verbose
        self._selectx = selectx
        self._test_size = test_size
        self._test_only = test_only
        self._split_method = split_method

    def build_json_datasets(self, in_folder, out_folder):
        if self._verbose:
            print(f'BUILDING JSON DATASET FOR {self._lang}\nin folder: {in_folder}\nout folder: {out_folder}')
        # new workflow:
        # keep in mind max. compatibility with the final format: spacy Docs with span annotations
        # 1. load all annotations in List[SpanAnnotation] format
        # 2. split into train and test, or test only
        # 2.1 split by text id - group by text id, split, then flatten
        # 3. for each split # how does this reflect on train/test data input to the model
        # 3.1 group by label
        # 3.2 for each label, create a .json dataset of texts in BIO format
        # 3.2.1 !!! new matching of annotations to tokens - using map_annotation_to_spacy_tokens()
        raw_authlabel = self.annot_load_group_by_authtextlabel(in_folder)
        # in new iteration, use per-file splitting: train, test = split_spans_by_text_id(raw_spans, test_size=self._test_size)
        if self._split_method == 'stratify':
            train, test = train_test_split(raw_authlabel, test_size=self._test_size, random_state=self._rseed,
                                           stratify=list(map(lambda ins: ins.label, raw_authlabel)))
        elif self._split_method == 'author-label':
            altag = lambda a: f'{a.author}-{a.label}'
            auth_lab = list(set(altag(a) for a in raw_authlabel))
            train_al, test_al = train_test_split(auth_lab, test_size=self._test_size, random_state=self._rseed,
                                           stratify=list(map(lambda al: al[-1], auth_lab)))
            print('train_al:', train_al)
            print('test_al:', test_al)
            train_al, test_al = set(train_al), set(test_al)
            train = [ s for s in raw_authlabel if altag(s) in train_al ]
            test = [s for s in raw_authlabel if altag(s) in test_al]
        splits = [train, test] if not self._test_only else [test]
        for split in splits:
            split_label = 'train' if split == train else 'test'
            print(f'SPLIT: {split_label}')
            by_label = group_by_perm(split, lambda inst: inst.label)
            # remove NONE_LABEL texts, and add a proportion of them (randomly) to all other labels' files
            none_label = self._ls_parser.none_label
            xfiles = list(by_label[none_label]) if none_label in by_label else []
            del by_label[none_label] # by_label.pop(NONE_LABEL)
            numx = math.ceil(len(xfiles)*self._selectx)
            std_labels = list(by_label.keys())
            print(std_labels)
            if numx > 0:
                for label in std_labels:
                    x_to_label = deepcopy(list(random.sample(xfiles, numx)))
                    for l in x_to_label: l.label = label
                    ext = list(by_label[label]); ext.extend(x_to_label)
                    by_label[label] = ext
            for label in std_labels:
                inst = by_label[label]
                # prepare output dir/file
                (Path(out_folder) / label).mkdir(exist_ok=True)
                split_outfile = Path(out_folder) / label / f'{split_label}.json'
                json_inst = self.instances_to_json(inst)
                with open(split_outfile, 'w') as fp:
                    json.dump(json_inst, fp)


    def annot_load_group_by_authtextlabel(self, folder) -> List[AuthLabelTxtSpans]:
        '''
        Create labeled instances in the 'annot' format, such that for each author and text a
        separate instance is created for each of the author's labels within the text.
        '''
        annots = ls_parse_all_subfolders(folder, self._ls_parser)
        print_annot_stats(annots)
        ann_byauthtxt = group_by_perm(annots, lambda a: (a.text_id, a.author))
        instances = []
        for txtauth, annots in ann_byauthtxt.items():
            txt_id, author = txtauth
            annotss = annots
            for label, lann in group_by_perm(annotss, lambda a: a.label).items():
                txt = lann[0].text
                insta = AuthLabelTxtSpans(label=label, text=txt, text_id=txt_id, author=author, spans=[])
                if label != self._ls_parser.none_label: # add spans, sorted by start
                    insta.spans = sorted([(ann.start, ann.end) for ann in lann], key=lambda s: s[0])
                    # check for cases of one span inside another, remove the inner one
                    again = True
                    while again: # repeat until no more changes
                        again = False
                        for i in range(len(insta.spans)-1):
                            s1, e1 = insta.spans[i]
                            s2, e2 = insta.spans[i+1]
                            if s2 >= s1 and e2 <= e1:
                                print(f'ANNOT. ERROR CORRECT: txt {txt_id} removing [{txt[s2:e2]}] leaving [{txt[s1:e1]}]')
                                insta.spans.pop(i+1)
                                again = True; break
                            elif s1 >= s2 and e2 <= e1:
                                print(f'ANNOT. ERROR CORRECT: txt {txt_id} removing [{txt[s1:e1]}] leaving [{txt[s2:e2]}]')
                                insta.spans.pop(i)
                                again = True; break
                instances.append(insta)
        instances.sort(key=lambda ins: ins.text_id)
        #for ins in instances: print(ins)
        return instances

    def _get_nlp(self):
        """ Return a spacy model for the language of the dataset. """
        #return spacy.blank(self._lang)
        if self._lang == 'en':
            #nlp = spacy.load('en_core_web_sm')
            nlp = spacy.blank('en')
        elif self._lang == 'es':
            #nlp = spacy.load('es_core_news_sm')
            nlp = spacy.blank('es')
        else: raise ValueError(f'Language {self._lang} not supported')
        nlp.tokenizer = SpacyURLTokenizer(nlp)
        return nlp

    def instances_to_json(self, inst: List[AuthLabelTxtSpans]):
        ''' Convert list of instances belonging to a same file,
         to a json file with tokens and labels in B-I-O sequence labeling format. '''
        labels = set([i.label for i in inst])
        assert len(labels) == 1
        #txt_ids = set([i.text_id for i in inst])
        #assert len(txt_ids) == 1
        nlp = self._get_nlp()
        label = inst[0].label
        BEGIN = f'B-{label}'
        INSIDE = f'I-{label}'
        OTHER = 'O'
        print(f'JSON generation for label {label}')
        num_span, num_none = 0, 0
        json_records = []
        for ins in inst:
            text = ins.text
            doc = nlp(text)
            # TODO do tokenization check
            #spacy_tok_check(doc, text)
            if self._verbose:
                print(f'TEXT: {ins.text_id}\n{text}')
                print('TOKENS: ', ';'.join([f'{tok.text};{tok.idx}-{tok.idx+len(tok)}' for tok in doc]))
            ins.spans.sort(key=lambda s: s[0])
            # convert ins (AuthLabelTxtSpans) to a list of SpanAnnotation objects
            span_annot = [SpanAnnotation(start=s[0], end=s[1], label=label,
                                         text=text, text_id=ins.text_id, author=ins.author)
                          for s in ins.spans]
            span2tokens = match_span_annots_to_spacy_spans_old(doc, span_annot)
            num_span += len(span_annot) # num. of spans
            num_none += len(span_annot) - len(span2tokens) # num. of spans not matched to tokens
            # print matching data
            print('-' * 30)
            for annot, tokens in span2tokens.items(): print(f'ANNOT: {annot}; TOKENS: {tokens}')
            unmatched = [s for s in span_annot if s not in span2tokens]
            if unmatched:
                print('UNMATCHED:')
                print(f'TEXT:\n{text}\n')
                for s in unmatched: print(s)
                print('-'*30)
            # create a list of tokens and labels in B-I-O format, based on the matching
            annot_tok_ranges = sorted([(toks[0].i, toks[-1].i) for toks in span2tokens.values()], key=lambda r: r[0])
            next_tok_range_ix = 0 if len(span2tokens) > 0 else None
            tokens, labels = [], []
            for tok in doc:
                tokens.append(tok.text)
                if next_tok_range_ix is not None:
                    ntr = annot_tok_ranges[next_tok_range_ix]
                    # if tok.i is in next token range
                    if tok.i >= ntr[0] and tok.i <= ntr[1]:
                        if tok.i == ntr[0]:
                            labels.append(BEGIN)
                        elif tok.i < ntr[1]:
                            labels.append(INSIDE)
                        else: # end of span
                            labels.append(INSIDE)
                            next_tok_range_ix += 1
                            if next_tok_range_ix >= len(annot_tok_ranges):
                                next_tok_range_ix = None
                    else:
                        labels.append(OTHER)
                else:
                    labels.append(OTHER)
            assert(len(tokens) == len(labels))
            lbl_str = ';'.join(labels)
            jrec = {LABELS_FIELD_NAME: labels, TOKENS_FIELD_NAME: tokens,
                    TEXT_ID_FIELD_NAME: ins.text_id, ORIG_GOLD_FIELD_NAME: lbl_str}
            if self._verbose:
                lens = [ max(len(l), len(t)) for l, t in zip(labels, tokens) ]
                N = len(tokens)
                print(' '.join(f'{tokens[i]:{lens[i]}}' for i in range(N)))
                print(' '.join(f'{labels[i]:{lens[i]}}' for i in range(N)))
                print()
            json_records.append(jrec)
        print(f'{num_span} spans, {num_none} missed, {num_none/num_span*100:.3f} perc')
        return json_records

class LS2HugginFaceLabelSep:
    '''
    Converts Label Studio annotations to the HuggingFace dataset format for sequence labeling.
    Creates a separate dataset for each of the label - in order to train a separate model for each label.
    '''

    def __init__(self, ls_parser: LabelStudioParser, lang, rseed=7120123, selectx=0.5,
                    test_size=0.25, test_only=False, split_method='stratify', verbose=False):
        set_random_seed(rseed)
        self._rseed = rseed
        self._ls_parser = ls_parser
        self._lang = lang
        self._verbose = verbose
        self._selectx = selectx
        self._test_size = test_size
        self._test_only = test_only
        self._split_method = split_method

    def build_json_datasets(self, in_folder, out_folder):
        #if self._verbose:
        print(f'BUILDING JSON DATASET FOR {self._lang}\nin folder: {in_folder}\nout folder: {out_folder}')
        raw_authlabel = self.annot_load_group_by_authtextlabel(in_folder)
        # in new iteration, use per-file splitting: train, test = split_spans_by_text_id(raw_spans, test_size=self._test_size)
        if self._split_method == 'stratify':
            train, test = train_test_split(raw_authlabel, test_size=self._test_size, random_state=self._rseed,
                                           stratify=list(map(lambda ins: ins.label, raw_authlabel)))
        elif self._split_method == 'author-label':
            altag = lambda a: f'{a.author}-{a.label}'
            auth_lab = list(set(altag(a) for a in raw_authlabel))
            train_al, test_al = train_test_split(auth_lab, test_size=self._test_size, random_state=self._rseed,
                                           stratify=list(map(lambda al: al[-1], auth_lab)))
            train_al, test_al = set(train_al), set(test_al)
            train = [ s for s in raw_authlabel if altag(s) in train_al ]
            test = [s for s in raw_authlabel if altag(s) in test_al]
        splits = [train, test] if not self._test_only else [test]
        for split in splits:
            split_label = 'train' if split == train else 'test'
            print(f'SPLIT: {split_label}')
            by_label = group_by_perm(split, lambda inst: inst.label)
            # remove NONE_LABEL texts, and add a proportion of them (randomly) to all other labels' files
            none_label = self._ls_parser.none_label
            xfiles = list(by_label[none_label]) if none_label in by_label else []
            del by_label[none_label] # by_label.pop(NONE_LABEL)
            numx = math.ceil(len(xfiles)*self._selectx)
            std_labels = list(by_label.keys())
            print(std_labels)
            if numx > 0:
                for label in std_labels:
                    x_to_label = deepcopy(list(random.sample(xfiles, numx)))
                    for l in x_to_label: l.label = label
                    ext = list(by_label[label]); ext.extend(x_to_label)
                    by_label[label] = ext
            for label in std_labels:
                inst = by_label[label]
                # prepare output dir/file
                (Path(out_folder) / label).mkdir(exist_ok=True)
                split_outfile = Path(out_folder) / label / f'{split_label}.json'
                json_inst = self.instances_to_json(inst)
                with open(split_outfile, 'w') as fp:
                    json.dump(json_inst, fp)


    def annot_load_group_by_authtextlabel(self, folder) -> List[AuthLabelTxtSpans]:
        '''
        Create labeled instances in the 'annot' format, such that for each author and text a
        separate instance is created for each of the author's labels within the text.
        '''
        annots = ls_parse_all_subfolders(folder, self._ls_parser)
        print_annot_stats(annots)
        ann_byauthtxt = group_by_perm(annots, lambda a: (a.text_id, a.author))
        instances = []
        for txtauth, annots in ann_byauthtxt.items():
            txt_id, author = txtauth
            annotss = annots
            for label, lann in group_by_perm(annotss, lambda a: a.label).items():
                txt = lann[0].text
                insta = AuthLabelTxtSpans(label=label, text=txt, text_id=txt_id, author=author, spans=[])
                if label != self._ls_parser.none_label: # add spans, sorted by start
                    insta.spans = sorted([(ann.start, ann.end) for ann in lann], key=lambda s: s[0])
                    # check for cases of one span inside another, remove the inner one
                    again = True
                    while again: # repeat until no more changes
                        again = False
                        for i in range(len(insta.spans)-1):
                            s1, e1 = insta.spans[i]
                            s2, e2 = insta.spans[i+1]
                            if s2 >= s1 and e2 <= e1:
                                print(f'ANNOT. ERROR CORRECT: txt {txt_id} removing [{txt[s2:e2]}] leaving [{txt[s1:e1]}]')
                                insta.spans.pop(i+1)
                                again = True; break
                            elif s1 >= s2 and e2 <= e1:
                                print(f'ANNOT. ERROR CORRECT: txt {txt_id} removing [{txt[s1:e1]}] leaving [{txt[s2:e2]}]')
                                insta.spans.pop(i)
                                again = True; break
                instances.append(insta)
        instances.sort(key=lambda ins: ins.text_id)
        #for ins in instances: print(ins)
        return instances

    def _get_nlp(self):
        """ Return a spacy model for the language of the dataset. """
        #return spacy.blank(self._lang)
        if self._lang == 'en':
            #nlp = spacy.load('en_core_web_sm')
            nlp = spacy.blank('en')
        elif self._lang == 'es':
            #nlp = spacy.load('es_core_news_sm')
            nlp = spacy.blank('es')
        else: raise ValueError(f'Language {self._lang} not supported')
        nlp.tokenizer = SpacyURLTokenizer(nlp)
        return nlp

    def instances_to_json(self, inst: List[AuthLabelTxtSpans]):
        ''' Convert list of instances belonging to a same file,
         to a json file with tokens and labels in B-I-O sequence labeling format. '''
        labels = set([i.label for i in inst])
        assert len(labels) == 1
        #txt_ids = set([i.text_id for i in inst])
        #assert len(txt_ids) == 1
        nlp = self._get_nlp()
        label = inst[0].label
        BEGIN = f'B-{label}'
        INSIDE = f'I-{label}'
        OTHER = 'O'
        print(f'JSON generation for label {label}')
        num_span, num_none = 0, 0
        json_records = []
        for ins in inst:
            text = ins.text
            doc = nlp(text)
            # TODO do tokenization check
            #spacy_tok_check(doc, text)
            if self._verbose:
                print(f'TEXT: {ins.text_id}\n{text}')
                print('TOKENS: ', ';'.join([f'{tok.text};{tok.idx}-{tok.idx+len(tok)}' for tok in doc]))
            tok_spans = []
            start_at = None
            ins.spans.sort(key=lambda s: s[0])
            prev_span = None
            current_token_index = 0
            for s in ins.spans:
                span_txt = text[s[0]: s[1]+1].strip()
                span_start, span_end = s[0], s[1]
                if self._verbose: print(f'SPAN: {span_txt}')
                if prev_span:
                    if prev_span[1] > s[0]:
                        if self._verbose: print(f'SPAN OVERLAP: {ins.text_id}')
                        continue
                        #assert (prev_span[1] <= s[0])
                if len(span_txt) == 0:
                    if self._verbose: print(f'WARNING, empty span in file: {ins.text_id}')
                    continue # empty span
                num_span += 1
                # find spacy tokens corresponding to span boundaries
                start_token = None
                end_token = None
                for token_index in range(current_token_index, len(doc)):
                    token = doc[token_index]
                    if token.idx == span_start:
                        start_token = token.i
                    if token.idx + len(token) == span_end:
                        end_token = token.i
                        current_token_index = token_index  # Update the current token index
                        break  # Break the loop once the end token is found
                if start_token is None or end_token is None:
                    num_none += 1
                    if self._verbose: print(f"Error: Span not found {span_start}-{span_end}, text: {span_txt}")
                else:
                    tok_span = (start_token, end_token + 1)
                    tok_spans.append(tok_span)
                    prev_span = tok_span
                # ts = get_token_span(doc, span_txt, soft=True, start_at=start_at)
                # if ts is None:
                #     # try again, by moving the border one char to the right
                #     span_txt = text[s[0]: s[1]+1].strip()
                #     ts = get_token_span(doc, span_txt, soft=True, start_at=start_at)
                #     if ts is None:
                #         if self._verbose: print(f'Missing span!')
                #         num_none += 1
                # else:
                #     tok_spans.append(ts)
                #     if self._verbose: print(f'Found span...')
                #     start_at = ts[1]
                #     prev_span = ts
            if tok_spans:
                tok_spans.sort(key=lambda ts: ts[0])
                tsi = 0
                s, e = tok_spans[tsi][0], tok_spans[tsi][1]
                started = False
                tokens = []
                labels = []
                for i, t in enumerate(doc):
                    tokens.append(t.text)
                    if not started:
                        if i < s: labels.append(OTHER)
                        elif i == s:
                            labels.append(BEGIN)
                            started = True
                    else:
                        if i < e: labels.append(INSIDE)
                        else:
                            started = False
                            tsi += 1
                            if tsi < len(tok_spans): # still more spans
                                if tok_spans[tsi][0] < e:
                                    print(ins.text_id)
                                    assert(tok_spans[tsi][0] >= e) # next span's start must be beyond the last's end
                                s, e = tok_spans[tsi][0], tok_spans[tsi][1]
                                if i == s: # check if next span imediately follows
                                    labels.append(BEGIN)
                                    started = True
                                else: labels.append(OTHER)
                            else:
                                s, e = len(doc)+1, None
                                labels.append(OTHER)

            else:
                tokens = [t.text for t in doc]
                labels = [OTHER] * len(tokens)
            assert(len(tokens) == len(labels))
            lbl_str = ';'.join(labels)
            jrec = {LABELS_FIELD_NAME: labels, TOKENS_FIELD_NAME: tokens,
                    TEXT_ID_FIELD_NAME: ins.text_id, ORIG_GOLD_FIELD_NAME: lbl_str}
            if self._verbose:
                lens = [ max(len(l), len(t)) for l, t in zip(labels, tokens) ]
                N = len(tokens)
                print(' '.join(f'{tokens[i]:{lens[i]}}' for i in range(N)))
                print(' '.join(f'{labels[i]:{lens[i]}}' for i in range(N)))
                print()
            json_records.append(jrec)
        print(f'{num_span} spans, {num_none} missed, {num_none/num_span*100:.3f} perc')
        return json_records


def test_loading_raw(dir=SEQLABEL_ANNOTS_EN_RAW):
    annots = ls_parse_all_subfolders(dir)
    grouped = group_annots_by_author_text(annots)
    for k, v in grouped.items():
        print(f'{k}: {v}')

def test_json_load(file):
    # dataset = load_dataset("json", data_files={"train": base_url + "train-v1.1.json", "validation": base_url + "dev-v1.1.json"}, field="data")
    dset = load_dataset("json", data_files=file)
    print(dset)

def load_as_hf_dataset(lang, label, split):
    if lang == 'en': dfolder = SEQLABEL_ANNOTS_EN_JSON
    else: dfolder = SEQLABEL_ANNOTS_ES_JSON
    features = {
        TEXT_ID_FIELD_NAME: datasets.Value("string"),
        ORIG_GOLD_FIELD_NAME: datasets.Value("string"),
        TOKENS_FIELD_NAME: datasets.Sequence(datasets.Value("string")),
        LABELS_FIELD_NAME: datasets.Sequence(
            datasets.features.ClassLabel(
                names=[
                    "O",
                    f"B-{label}",
                    f"I-{label}",
                ]
            )
        ),
    }
    features = datasets.Features(features)
    dset = load_dataset('json', data_files=str(Path(dfolder) / label / f'{split}.json'), features=features)
    return dset

def build_json_datasets(lang='en', rseed=348461, selectx=0.5, test_only=False, split_method='stratify',
                        test_size=0.25, verbose=False):
    if lang == 'en':
        input_folder = SEQLABEL_ANNOTS_EN_RAW
        #out_folder = SEQLABEL_ANNOTS_EN_JSON
        out_folder = SEQLABEL_ANNOTS_EN_JSON_DEV
    else:
        input_folder = SEQLABEL_ANNOTS_ES_RAW
        out_folder = SEQLABEL_ANNOTS_ES_JSON
    ls_parser = LabelStudioParser()
    ls2hf = LS2HugginFaceLabelSepNew(ls_parser=ls_parser, lang=lang, selectx=selectx, test_only=test_only,
                                     split_method=split_method, test_size=test_size, verbose=verbose, rseed=rseed)
    ls2hf.build_json_datasets(input_folder, out_folder)

if __name__ == '__main__':
    #parse_json_annotations(SEQLABEL_ANNOTS_EN_RAW)
    #test_loading_raw()
    #load_annotations_seplabel(SEQLABEL_ANNOTS_EN)
    #test_json_load('/data/corpora/UPV/xai-disinfo-seqlabel/json/en/V/train.json')
    #build_json_datasets('/data/corpora/UPV/xai-disinfo-seqlabel/json/es/', 'es')
    #build_json_datasets(SEQLABEL_ANNOTS_EN_JSON, 'en', test_only=True)
    build_json_datasets('en', test_only=False, verbose=False,
                        split_method='stratify', rseed=515, test_size=0.3)
    #build_json_datasets(SEQLABEL_ANNOTS_ES_JSON, 'es', test_only=False, split_method='author-label')
