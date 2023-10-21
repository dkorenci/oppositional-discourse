'''

Methods for loading and preparing the sequence labeling data.

'''

import json
import math
import traceback
from copy import copy, deepcopy
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

from sequence_labeling.article_example import get_token_span
from data_tools.seqlab_data.data_struct import AuthLabelTxtSpans
from settings import *

PER_FILE_OUT_FOLDER = 'per-file-data'

warnings.filterwarnings("ignore")

SHORT_LABELS = None
SIM_PAIRS = None
LABEL_GROUPS = None

NONE_LABEL = 'X' # must not be equal to any abbreviation

SHORT_LABELS_V2 = {
    'OBJETIVOS': 'O', 'AGENTE': 'A', 'FACILITADORES': 'F',
    'PARTIDARIOS': 'P', 'VÍCTIMAS': 'V', 'EFECTOS_NEGATIVOS': 'E',
}
SHORT_LABELS_SIM_PAIRS_V2 = [ ('A', 'F'), ('V', 'P'), ('O', 'E') ]

LABEL_GROUPS_V2 = [
    ['AGENTE', 'EFECTOS_NEGATIVOS', 'FACILITADORES',
     'PARTIDARIOS', 'VÍCTIMAS'],
    ['OBJETIVOS', ]
]

SHORT_LABELS_V1 = {
    'AGENTE':'A', 'ESTRATEGIAS_PERSUASIÓN':'E', 'FACILITADORES':'F',
    'PARTIDARIOS':'P', 'RELATO':'R', 'VÍCTIMAS': 'V'
}
SHORT_LABELS_SIM_PAIRS_V1 = [ ('A', 'F'), ('V', 'P'), ('R', 'E') ]

LABEL_GROUPS_V1 = [
    ['AGENTE', 'ESTRATEGIAS_PERSUASIÓN', 'FACILITADORES',
     'PARTIDARIOS', 'VÍCTIMAS'],
    ['RELATO', ]
]

# fields that hold binary labeling information in json record 'data' field
CONSPI_FIELDS = ['GS_CONSPI_THEORY', 'GS_CONSPI-THEORY']
CT_FIELDS = ['GS_CRITICAL_THINKING']
def get_label_field(d, fields):
    ''' Extract the value of one of the fields from d, as int. '''
    for f in fields:
        if f in d: return int(d[f])
    return None

def extract_annot(jprs, annotator):
    ''' Extract annotations from annotations created using Labels Studio's json export.
     Annotations correspond to one text document, annotatesd by one annotator.  '''
    result = []
    for r in jprs:
        #print(r['file_upload'], r['data']['id'])
        data = r['data']
        file_id, text, file_num = data['id'], data['body'], int(data['NUM'])
        # ignore text if it contains no critical thinking and no conspiracy theory
        conspi = get_label_field(data, CONSPI_FIELDS)
        critic = get_label_field(data, CT_FIELDS)
        if conspi is not None and critic is not None:
            if conspi == 0 and critic == 0:
                print(f'WARNING: 0-0 labels for text {file_id}, ignoring')
                continue
        else: print(f'Nonexistent or partial GS metadata in {file_id}')
        annots = r['annotations']
        if len(annots) != 1:
            print(f'WARNING: length of "annotations" block is not 1 but {len(annots)}\nANNOTS:{annots}')
        spans = annots[0]['result']
        if len(spans) == 0: # no spans at all for this annotator, add special annotation
            print(f'NO SPANS for the file {file_id}, annotator: {annotator}')
            result.append({'file_id': file_id, 'file_num': file_num, 'text': text, 'label': NONE_LABEL})
        else:
            startend_mismatch = False; empty_span = False
            for span in spans:
                # read span data
                assert span['type'] == 'hypertextlabels'
                spdata = span['value']
                start, end = spdata['startOffset'], spdata['endOffset']
                start_global, end_global = spdata['globalOffsets']['start'], spdata['globalOffsets']['end']
                # check for span boundaries errors and warnings
                if start != start_global or end != end_global: startend_mismatch = True
                if start_global == end_global:
                    empty_span = True
                    continue # ignore empty spans
                elif start_global > end_global:
                    raise ValueError(f'Span start ({start_global}) > end ({end_global}) in file {file_id}')
                # get label data and create an annotation record
                labels = spdata['hypertextlabels']
                assert len(labels) == 1
                label = labels[0]
                result.append({'file_id': file_id, 'file_num': file_num, 'text': text,
                               'label': label, 'start': start_global, 'end': end_global})
            if startend_mismatch: print(f'Offsets and global offsets mismatch in text {file_id}')
            if empty_span: print(f'WARNING, start == end for a span in text {file_id}')
    return result

def parse_file(pth):
    '''
    Parse a json file with Label Studio annotations.
    Asumes author per file, author name embeded in file name.
    '''
    try:
        name = pth.name.split('.')[-2].split('_')[-1]
    except:
        raise Exception(f'Error, file name must end with _AUTHORNAME.json. File name: {pth.name}')
    parse = json.load(open(pth, 'r'))
    annots = extract_annot(parse, name)
    for a in annots: a['author'] = name
    return annots

def parse_folder(folder, print_labels=True):
    '''
    Parse annotation files in a folder.
    '''
    annots = []
    unique_labels = lambda annots: sorted(set(ann['label'] for ann in annots))
    for pth in Path(folder).glob('*.json'):
        print(pth)
        file_annots = parse_file(pth)
        print('ALL FILE LABELS: ', unique_labels(file_annots))
        annots.extend(file_annots)
    if print_labels: print('ALL LABELS:', unique_labels(annots))
    return annots

def complete_annotations(annots, all_authors, file_id, text):
    complete = []
    authors_none = []
    num_annot = 0
    authors = []
    for ann in annots:
        annotator, label = ann['author'], ann['label']
        assert(ann['file_id'] == file_id)
        if label != NONE_LABEL: # standard annotation
            if label in SHORT_LABELS: label = SHORT_LABELS[label]
            else: print(f'WARNING: NO ABBREVIATION FOR LABEL: {label}')
            annc = copy(ann)
            annc['label'] = label
            complete.append(annc)
            authors.append(annotator)
            num_annot += 1
        else: # special label indicating that the annotator has no annotations for the file
            authors_none.append(annotator)
    if num_annot == 0: print('WARNING: no annotations in the file')
    assert len(authors_none) == len(set(authors_none)) # check that each 'none' author occurs only once
    authors_none = set(authors_none)
    authors = set(authors)
    # authors that haven't got the file_id in their .json
    # so they aren't present but labeled with NONE_LABEL
    authors_notinfile = set(all_authors).difference(authors.union(authors_none))
    all_empty = authors_none.union(authors_notinfile)
    if len(all_empty) :
        print(f'Warning: "NONE" author occurr: {authors_none.union(authors_notinfile)}')
        print(f'    added authors: {authors}')
        for a in all_empty:
            complete.append({'file_id': file_id, 'text': text, 'label': NONE_LABEL, 'author': a})
    return complete

def parse_lang_from_folder_name(folder):
    folder = PurePath(folder).name
    if '-en-' in folder: return 'en'
    elif '-es-' in folder: return 'es'
    else: raise ValueError(f'Cannot extract language from folder name: {folder}')

def group_annots(annots, select):
    res = []
    for k, g in groupby(annots, select):
        res.append((k, list(g)))
    return res

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

def set_labels(label_version):
    global SHORT_LABELS, SIM_PAIRS, LABEL_GROUPS
    if label_version == 1:
        SHORT_LABELS, SIM_PAIRS = SHORT_LABELS_V1, SHORT_LABELS_SIM_PAIRS_V1
        LABEL_GROUPS = LABEL_GROUPS_V1
    elif label_version == 2:
        SHORT_LABELS, SIM_PAIRS = SHORT_LABELS_V2, SHORT_LABELS_SIM_PAIRS_V2
        LABEL_GROUPS = LABEL_GROUPS_V2
    else: raise ValueError(f'Unsupported label version: {label_version}')

def check_file_nums(annots_byfile):
    ''' Check if the same file_id texts have the same file_num. '''
    numsg = set()
    print(f'Number of texts: {len(annots_byfile)}')
    for file_id, annots in annots_byfile:
        numsl = set(an['file_num'] for an in annots)
        if len(numsl) != 1:
            raise ValueError(f'for text {file_id} different file_nums occurr across annotators: {numsl}')
        file_num = list(numsl)[0]
        if file_num in numsg:
            raise ValueError(f'file_num {file_num} occurrs for more than on file_id')
        else:
            numsg.add(file_num)
    return annots

def analyze_folder_gamma(folder, label_version=2):
    '''
    Parse annotation files in a folder and do analysis based on gamma IAA metric.
    :param folder:
    :return:
    '''
    set_labels(label_version)
    print('Analyzing folder: ', folder)
    print('Parsing the files ...')
    annots = parse_folder(folder)
    all_authors = set(a['author'] for a in annots)
    file_select = lambda x: x['file_id'] # file id from annot. record
    annots.sort(key=file_select)
    annots_byfile = group_annots(annots, file_select)
    check_file_nums(annots_byfile)
    print('Completing gammas ...\n')
    fid2txt = {}
    for ann in annots:
        fid, txt = ann['file_id'], ann['text']
        if fid not in fid2txt: fid2txt[fid] = txt
        else:
            assert txt == fid2txt[fid]
    result = []
    for file_id, annots in annots_byfile:
        try:
            ann_list = [ann for ann in annots]
            complete_annots = complete_annotations(ann_list, all_authors=all_authors, file_id=file_id, text=fid2txt[file_id])
            result.extend(complete_annots)
        except Exception as e:
            etxt = f'ERROR for file_id: {file_id} - {e}'
            print(etxt)
            print(traceback.format_exc())
    return result

def parse_json_annotations(folder):
    '''
    Parse all the annotations for the folder containing a subfolder per batch,
    containing the per-annotator files.
    :return: list of dicts with
    '''
    #return [ ann for f in Path(folder).glob('*') for ann in parse_folder(f) ]
    return [ann for f in Path(folder).glob('*') for ann in analyze_folder_gamma(f)]

def group_annots_by_author_text(annots):
    keyf = lambda a: (a['file_id'], a['author'])
    annots = sorted(annots, key=keyf)
    gb = groupby(annots, keyf)
    res = { k: list(g) for k, g in gb }
    return res

def print_annot_stats(annots):
    by_txt = group_by_perm(annots, lambda a: (a['file_id']))
    num_txt = len(by_txt)
    print(f'Total number of annotated texts: {num_txt}')
    N = len(annots)
    print(f'Total number of annotations: {N}')
    print(f' Avg. number of annotations per text: {N / num_txt:.3f}')

def load_annotations_seplabel(folder) -> List[AuthLabelTxtSpans]:
    '''
    Create labeled instances in the 'annot' format, such that for each author and text a
    separate instance is created for each of the author's labels within the text.
    '''
    annots = parse_json_annotations(folder)
    print_annot_stats(annots)
    ann_byauthtxt = group_by_perm(annots, lambda a: (a['file_id'], a['author']))
    instances = []
    for txtauth, annots in ann_byauthtxt.items():
        txt_id, author = txtauth
        #akey = lambda a: a['label']
        #annotss = sorted(annots, key=akey)
        annotss = annots
        for label, lann in group_by_perm(annotss, lambda a: a['label']).items(): #groupby(annotss, akey):
            txt = lann[0]['text']
            insta = AuthLabelTxtSpans(label=label, text=txt, text_id=txt_id, author=author, spans=[])
            if label != NONE_LABEL: # add spans, sorted by start
                insta.spans = sorted([(ann['start'], ann['end']) for ann in lann], key=lambda s: s[0])
            instances.append(insta)
    instances.sort(key=lambda ins: ins.text_id)
    #for ins in instances: print(ins)
    return instances

def group_by_perm(data, keyf):
    keys = set([keyf(d) for d in data])
    res = { k: [] for k in keys }
    for d in data: res[keyf(d)].append(copy(d))
    for k, itms in res.items():
        keys = set([keyf(it) for it in itms])
        assert len(keys) == 1
        assert keys.pop() == k
    return res

TOKENS_FIELD_NAME = 'words'
LABELS_FIELD_NAME = 'narrative_tag'
TEXT_ID_FIELD_NAME = 'text_id'
ORIG_GOLD_FIELD_NAME = 'orig_gold'

def instances_to_json(inst: List[AuthLabelTxtSpans], lang, verbose=False):
    ''' Convert list of instances belonging to a same file,
     to a json file with tokens and labels in B-I-O sequence labeling format. '''
    labels = set([i.label for i in inst])
    assert len(labels) == 1
    #txt_ids = set([i.text_id for i in inst])
    #assert len(txt_ids) == 1
    nlp = spacy.blank(lang)
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
        if verbose:
            print(f'TEXT: {text}')
            print('TOKENS: ', ';'.join([tok.text for tok in doc]))
        tok_spans = []
        start_at = None
        ins.spans.sort(key=lambda s: s[0])
        prev_span = None
        for s in ins.spans:
            span_txt = text[s[0]: s[1]].strip()
            if verbose: print(f'SPAN: {span_txt}')
            if prev_span:
                if prev_span[1] > s[0]:
                    print(f'SPAN OVERLAP: {ins.text_id}')
                    continue
                    #assert (prev_span[1] <= s[0])
            if len(span_txt) == 0:
                print(f'WARNING, empty span in file: {ins.text_id}')
                continue # empty span
            num_span += 1
            ts = get_token_span(doc, span_txt, soft=True, start_at=start_at)
            if ts is None:
                # try again, by moving the border one char to the right
                span_txt = text[s[0]: s[1]+1].strip()
                ts = get_token_span(doc, span_txt, soft=True, start_at=start_at)
                if ts is None:
                    if verbose: print(f'Missing span!')
                    num_none += 1
            else:
                tok_spans.append(ts)
                if verbose: print(f'Found span...')
                start_at = ts[1]
                prev_span = ts
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
        if verbose:
            lens = [ max(len(l), len(t)) for l, t in zip(labels, tokens) ]
            N = len(tokens)
            print(' '.join(f'{tokens[i]:{lens[i]}}' for i in range(N)))
            print(' '.join(f'{labels[i]:{lens[i]}}' for i in range(N)))
            print()
        json_records.append(jrec)
    print(f'{num_span} spans, {num_none} missed, {num_none/num_span*100:.3f} perc')
    return json_records

def build_json_datasets(out_folder, lang='en', rseed=348461, selectx=0.5, test_only=False, split_method='stratify',
                        test_size=0.25, verbose=False):
    print(f'BUILDING JSON DATASET FOR {lang}, out folder {out_folder}')
    set_random_seed(rseed)
    if lang == 'en': in_folder = SEQLABEL_ANNOTS_EN_RAW
    elif lang == 'es': in_folder = SEQLABEL_ANNOTS_ES_RAW
    raw_authlabel = load_annotations_seplabel(in_folder)
    if split_method == 'stratify':
        train, test = train_test_split(raw_authlabel, test_size=test_size, random_state=rseed,
                                       stratify=list(map(lambda ins: ins.label, raw_authlabel)))
    elif split_method == 'author-label':
        altag = lambda a: f'{a.author}-{a.label}'
        auth_lab = list(set(altag(a) for a in raw_authlabel))
        train_al, test_al = train_test_split(auth_lab, test_size=test_size, random_state=rseed,
                                       stratify=list(map(lambda al: al[-1], auth_lab)))
        print('train_al:', train_al)
        print('test_al:', test_al)
        train_al, test_al = set(train_al), set(test_al)
        train = [ s for s in raw_authlabel if altag(s) in train_al ]
        test = [s for s in raw_authlabel if altag(s) in test_al]
    splits = [train, test] if not test_only else [test]
    for split in splits:
        split_label = 'train' if split == train else 'test'
        print(f'SPLIT: {split_label}')
        by_label = group_by_perm(split, lambda inst: inst.label)
        # remove NONE_LABEL texts, and add a proportion of them (randomly) to all other labels' files
        xfiles = list(by_label[NONE_LABEL]) if NONE_LABEL in by_label else []
        del by_label[NONE_LABEL] # by_label.pop(NONE_LABEL)
        numx = math.ceil(len(xfiles)*selectx)
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
            json_inst = instances_to_json(inst, lang, verbose=verbose)
            with open(split_outfile, 'w') as fp:
                json.dump(json_inst, fp)

def test_loading_raw(dir=SEQLABEL_ANNOTS_EN_RAW):
    annots = parse_json_annotations(dir)
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

if __name__ == '__main__':
    build_json_datasets(SEQLABEL_ANNOTS_EN_JSON_TEST, 'en', test_only=False, verbose=True,
                        split_method='author-label', rseed=515, test_size=0.35)
    #build_json_datasets(SEQLABEL_ANNOTS_ES_JSON, 'es', test_only=False, split_method='author-label')
