import json, shutil, pandas as pd
import os
import traceback
from itertools import groupby
from pathlib import Path, PurePath
import random

import numpy as np
from pygamma_agreement import Continuum, CorpusShufflingTool, ShuffleContinuumSampler, StatisticalContinuumSampler, \
    PrecomputedCategoricalDissimilarity
from pyannote.core import Segment
from pygamma_agreement import CombinedCategoricalDissimilarity
from pygamma_agreement import show_alignment
import matplotlib.pyplot as plt
from pygamma_agreement.notebook import notebook
import argparse

import warnings

from sortedcontainers import SortedSet

from gamma_iaa.custom_continuum import ContinuumFlexSample
from gamma_iaa.custom_samplers import StatSamplerPerAnnotator, StatSamplerPerAnnotatorNumunits
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

def get_precomputed_categ_dissim(categs, sim_pairs, delta, empty=True):
    if empty: categs = list(categs) + [NONE_LABEL]
    N = len(categs);
    assert N == len(set(categs))  # check uniqueness
    D = np.ones((N, N))
    for i in range(N): D[i, i] = 0  # dissim. of categ. to self is 0
    categories = SortedSet(categs)
    for c1, c2 in sim_pairs: # for similar pairs, set dissim. to 0.5
        ix1, ix2 = categories.index(c1), categories.index(c2)
        D[ix1, ix2], D[ix2, ix1] = 0.5, 0.5
    # penalize heavility for the NONE_LABEL, to force alignment with 'empty' spans
    ei = categories.index(NONE_LABEL)
    for i in range(N):
        D[ei, i], D[i, ei] = 10, 10
    D[ei, ei] = 0
    return PrecomputedCategoricalDissimilarity(categories, matrix=D, delta_empty=delta)

def construct_dissimilarity(typ, alpha, beta, delta):
    if typ == 'standard':
        return CombinedCategoricalDissimilarity(alpha=alpha, beta=beta, delta_empty=delta)
    elif typ == 'custom_categ':
        cat_dissim = get_precomputed_categ_dissim(list(SHORT_LABELS.values()), SIM_PAIRS, delta)
        return CombinedCategoricalDissimilarity(cat_dissim=cat_dissim,
                                                alpha=alpha, beta=beta, delta_empty=delta)

def calc_gamma(annots, num_align, global_continuum=None, all_authors=None):
    continuum = ContinuumFlexSample()
    authors_none = []
    num_annot = 0
    max_pos = None
    authors = []
    for ann in annots:
        annotator, label = ann['author'], ann['label']
        if label != NONE_LABEL: # standard annotation
            start, end = ann['start'], ann['end']
            max_pos = end if max_pos is None else max(max_pos, end)
            if label in SHORT_LABELS: label = SHORT_LABELS[label]
            else: print(f'WARNING: NO ABBREVIATION FOR LABEL: {label}')
            #if (start != end): continuum.add(annotator, Segment(start, end), label)
            continuum.add(annotator, Segment(start, end+1), label)
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
        empty_len, empty_dist = 5, 10
        max_pos = 1 if max_pos is None else max_pos + empty_dist # start from beyond the end
        #raise Exception(f'Less then three authors, authors: {authors}')
        print(f'Warning: "NONE" author occurr: {authors_none.union(authors_notinfile)}')
        print(f'    added authors: {authors}')
        # create 'NONE' labels, all at the same position because they match mutually
        #   as two 'empty' annotators are in agreement
        for a in all_empty:
            continuum.add(a, Segment(max_pos, max_pos+empty_len), NONE_LABEL)
    dissim = construct_dissimilarity(typ='custom_categ', alpha=1.0, beta=1.0, delta=1.0)
    if global_continuum:
        sampler = StatSamplerPerAnnotator()
        # sampler = StatSamplerPerAnnotatorNumunits()
        sampler.init_sampling(reference_continuum=global_continuum)
        init_sampler = False
    else:
        sampler = None
        init_sampler = True
    gamma_results = continuum.compute_gamma(dissim, n_samples=num_align, sampler=sampler, init_sampler=init_sampler)
    return gamma_results, continuum

def generate_gamma_report(folder, gammas, errors, graphic_out=False):
    '''
    Generate gamma stats and graphs for a set of texts with
    associated gamma results from pygamma_agreement.
    :param folder: output folder
    :param gammas: map of { text_id: GammaResults}
    :return:
    '''
    report = open(Path(folder)/'gamma_report.txt', 'w')
    if graphic_out:
        graph_folder = Path(folder) / PER_FILE_OUT_FOLDER
    gstat = pd.Series([ res.gamma for _, res in gammas.items() ]).describe()
    report.write(f'Average gamma: {gstat["mean"]:.3f}\n\n')
    report.write('Distribution of gammas over texts:\n')
    dist_str = '; '.join(f'{stat}: {gstat[stat]:.3f}' for stat in gstat.describe().keys())
    report.write(dist_str+'\n')
    if errors:
        report.write('\nERRORS:\n')
        for e in errors: report.write(e+'\n')
        report.write('\n')
    report.write('Per-text gamma:\n')
    lang = parse_lang_from_folder_name(folder)
    for ids, gamma_results in sorted(gammas.items(), key=lambda g: g[1].gamma):
        file_id, file_num = ids
        report.write(f'file: {lang+"_"+str(file_num):8}, id: {file_id:20}, gamma: {gamma_results.gamma:.3f}\n')
        if graphic_out:
            w, h, my_dpi = 1928, 1272, 300
            fig, ax = plt.subplots(figsize=(w/my_dpi,h/my_dpi), dpi=my_dpi) #plt.subplots()
            notebook.plot_alignment_continuum(gamma_results.best_alignment, ax=ax, labelled=True)
            plt.tight_layout()
            outfile = graph_folder/f'{lang}_{file_num}_{file_id}.pdf'
            plt.savefig(outfile, dpi=my_dpi)
    report.close()
    return gstat['mean']

def parse_lang_from_folder_name(folder):
    folder = PurePath(folder).name
    if '-en-' in folder: return 'en'
    elif '-es-' in folder: return 'es'
    else: raise ValueError(f'Cannot extract language from folder name: {folder}')

def print_annots_old(fout, annots):
    ''' Print a group of annotations belonging to a same text into a file. '''
    txt = annots[0]['text']
    file_id = annots[0]['file_id']
    annot_sort = lambda a: f'{a["author"]}-{a["start"]:4}-{a["end"]:4}'
    prev_author = None
    print('TEXT:', file=fout)
    print(f'{txt}\n', file=fout)
    for ann in sorted(annots, key=annot_sort):
        assert ann['file_id'] == file_id
        label, start, end, author = ann['label'], int(ann['start']), int(ann['end']), ann['author']
        if prev_author and prev_author != author: print(file=fout)
        span_txt = txt[start:end]
        annot_data = f'{ann["author"]}, label: {label}, from {start} - {end} \ntext: {span_txt}'
        print(annot_data, file=fout)
        prev_author = author
    print(file=fout)

def print_annots_cont(fout, annots, continuum):
    ''' Print a group of annotations belonging to a same text into a file.
    Order by continuum order, to align it with the graphics. '''
    #for annot_id, annotator in enumerate(continuum.annotators):
    # order of annotators same as in the graphics display
    annot_order = [annot for _, annot in enumerate(continuum.annotators)]
    annot_order.reverse()
    txt = annots[0]['text']
    print('TEXT:', file=fout)
    print(f'{txt}\n', file=fout)
    by_annot = { annot: list(group) for annot, group in groupby(annots, lambda a: a['author']) }
    # add empty groups
    none_annots = set(a['author'] for a in annots if a['label'] == NONE_LABEL)
    file_annots = set(a['author'] for a in annots)
    cont_annots = set(annot_order)
    # add empty groups for annotators with NONE_LABEL
    for none_annot in none_annots: by_annot[none_annot] = []
    # add empty groups for annotators with files missing in .json (added to continuum before as empty, but no in read annots)
    for none_annot in cont_annots.difference(file_annots): by_annot[none_annot] = []
    # print labels grouped by this partitioning
    ann_by_pos = lambda a: f'{a["start"]:4}-{a["end"]:4}'
    for lgroup in LABEL_GROUPS:
        for annot in annot_order: # by annotator
            print(f'{annot}:', file=fout)
            for ann in sorted(by_annot[annot], key=ann_by_pos): # annotations sorted by position
                label = ann['label']
                if label in lgroup:
                    start, end = int(ann['start']), int(ann['end'])
                    span_txt = txt[start:end+1]
                    print(f'{label}: {span_txt}', file=fout)
            print(file=fout)

def print_file_annots(folder, annots, continuum):
    '''
    Print annotations of a single file, in a txt format.
    :param folder: folder to store the output
    :param annots: list of annotation data dicts, all with the same 'file_id'
    :return:
    '''
    lang = parse_lang_from_folder_name(folder)
    file_id = annots[0]['file_id']
    file_num = annots[0]['file_num']
    # parse input and create output
    with open(Path(folder)/PER_FILE_OUT_FOLDER/f'{lang}_{file_num}_{file_id}.txt', 'w') as fout:
        #print_annots_old(fout, annots)
        print_annots_cont(fout, annots, continuum)

def build_continuum(annots, per_file_annots=True):
    continuum = Continuum()
    for ann in annots:
        if ann['label'] != NONE_LABEL: # ignore empty set of annotations
            annotator, label, start, end = ann['author'], ann['label'], ann['start'], ann['end']
            if per_file_annots:
                file_num = ann['file_num']
                annotator = f'{annotator}_{file_num}'
            label = SHORT_LABELS.get(label, label)
            continuum.add(annotator, Segment(start, end), label)
    return continuum

# version with removing annotations from files with 1 annotator, just in case
# def build_continuum(annots, per_file_annots=True):
#     continuum = Continuum()
#     # remove files with less than 2 non-empty annotators
#     rem_file = []
#     for fid, group in groupby(annots, lambda a: a['file_id']):
#         num_annots = len(set(ann['author'] for ann in group if ann['label'] != NONE_LABEL))
#         if num_annots < 2: rem_file.append(fid)
#     rem_file = set(rem_file)
#     print('Global continuum, removed file ids: ', rem_file)
#     for ann in annots:
#         label, file_id = ann['label'], ann['file_id']
#         if file_id not in rem_file and ann['label'] != NONE_LABEL: # ignore empty set of annotations
#             annotator, start, end = ann['author'], ann['start'], ann['end']
#             if per_file_annots:
#                 file_num = ann['file_num']
#                 annotator = f'{annotator}_{file_num}'
#             label = SHORT_LABELS.get(label, label)
#             continuum.add(annotator, Segment(start, end), label)
#     return continuum

def group_annots(annots, select):
    res = []
    for k, g in groupby(annots, select):
        res.append((k, list(g)))
    return res

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

def test_global_sampling(folder, method='stat', rnd_seed=175492, num_samples=10, graphic_out=True):
    set_random_seed(rnd_seed)
    print('Analyzing folder: ', folder)
    print('Parsing the files ...')
    annots = parse_folder(folder)
    file_select = lambda x: x['file_id'] # file id from annot. record
    annots.sort(key=file_select)
    annots_byfile = group_annots(annots, file_select)
    check_file_nums(annots_byfile)
    continuum = build_continuum(annots, per_file_annots=True)
    #sampler = CorpusShufflingTool(magnitude=0.1, reference_continuum=continuum)
    if method == 'stat':
        sampler = StatSamplerPerAnnotator()
        #sampler = StatSamplerPerAnnotatorNumunits()
    elif method == 'shuff': sampler = ShuffleContinuumSampler()
    else: raise ValueError(f'unknown sampling method: {method}')
    sampler.init_sampling(reference_continuum=continuum)
    if isinstance(sampler, StatSamplerPerAnnotator):
        sampler.print_perannot_stats()
    for i in range(num_samples):
        cont = sampler.sample_from_continuum
        if graphic_out:
            w, h, my_dpi = 1928, 1272, 300
            fig, ax = plt.subplots(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)  # plt.subplots()
            notebook.plot_continuum(cont, ax=ax, labelled=True)
            plt.tight_layout()
            outfile = f'continuum_{i}.pdf'
            plt.savefig(outfile, dpi=my_dpi)


def set_labels(label_version):
    global SHORT_LABELS, SIM_PAIRS, LABEL_GROUPS
    if label_version == 1:
        SHORT_LABELS, SIM_PAIRS = SHORT_LABELS_V1, SHORT_LABELS_SIM_PAIRS_V1
        LABEL_GROUPS = LABEL_GROUPS_V1
    elif label_version == 2:
        SHORT_LABELS, SIM_PAIRS = SHORT_LABELS_V2, SHORT_LABELS_SIM_PAIRS_V2
        LABEL_GROUPS = LABEL_GROUPS_V2
    else: raise ValueError(f'Unsupported label version: {label_version}')

def analyze_folder_gamma(folder, rnd_seed=175492, num_align=100, graphic_out=True, use_global=False, label_version=1):
    '''
    Parse annotation files in a folder and do analysis based on gamma IAA metric.
    :param folder:
    :return:
    '''
    set_random_seed(rnd_seed)
    set_labels(label_version)
    print('Analyzing folder: ', folder)
    print('Parsing the files ...')
    annots = parse_folder(folder)
    all_authors = set(a['author'] for a in annots)
    file_select = lambda x: x['file_id'] # file id from annot. record
    annots.sort(key=file_select)
    annots_byfile = group_annots(annots, file_select)
    check_file_nums(annots_byfile)
    if use_global: folder_continuum = build_continuum(annots, per_file_annots=True)
    else: folder_continuum = None
    print('Calculating gammas ...\n')
    gammas = {}
    errors = []
    # prepare output folder
    per_file_data_fold = Path(folder) / PER_FILE_OUT_FOLDER
    if per_file_data_fold.exists(): shutil.rmtree(per_file_data_fold)
    per_file_data_fold.mkdir(exist_ok=True)
    for file_id, annots in annots_byfile:
        try:
            file_num = annots[0]['file_num']
            ann_list = [ann for ann in annots]
            gamma_results, continuum = calc_gamma(ann_list, num_align, folder_continuum, all_authors=all_authors)
            print_file_annots(folder, ann_list, continuum)
            gammas[(file_id, file_num)] = gamma_results
        except Exception as e:
            etxt = f'ERROR for file_id: {file_id} - {e}'
            errors.append(etxt)
            print(etxt)
            print(traceback.format_exc())
    # create and output analysis
    avg_gamma = generate_gamma_report(folder, gammas, errors, graphic_out)
    return avg_gamma

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

def run_cmdline():
    parser = argparse.ArgumentParser(
        description='''
        Create an gamma IAA analysis from a set of Label Studio .json files.
        Input is a folder with the .json files. 
        IMPORTANT: file names in the folder must end with _AUTHORNAME.json. 
        The program will output the statistics of the per-file gamma IAAs in gamma_report.txt, 
        and optionally the visualizations of the annotators's alignments.  
        All the output files will be stored in the input folder. 
        On commmand line the program can spam with a lot of internal warnings and messages.
        '''
    )
    parser.add_argument('-f', '--folder',
                        help='Path to the folder with annotation files',
                        required=True,
                        default='.',
                        type=str)
    parser.add_argument('-g', '--graphic_out',
                        help='Optional, create visualization of per-author annotations, one per file.',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-s', '--rnd_seed',
                        required=False,
                        default=175492,
                        help='Random seed for the gamma computation algorithm.',
                        type=int)
    parser.add_argument('-n', '--num_align',
                        required=False,
                        default=100,
                        help='Number of random alignments generated in order to assess the quality of the best alignment.'
                             'Increasing increases the quality of the gamma assessment but slows down the computation.',
                        type=int)
    args = vars(parser.parse_args())
    analyze_folder_gamma(**args)

def analyze_fase2new(use_global=False, num_align=100, label_version=1, range=None, folders=None):
    gammas, files = [], []
    if range is not None: # input is a start and end subfolder of a list ANNOT_FASE2NEW_FOLDERS
        all_folders = folders
        folders = []
        start, stop = False, False
        for i, f in enumerate(all_folders):
            if f.endswith(range[0]): start = True
            if f.endswith(range[1]): stop = True
            if not start: continue
            folders.append(f)
            if stop: break
    else: pass # input is a list of subfolders
    for f in folders:
        avg_gamma = analyze_folder_gamma(f, use_global=use_global, num_align=num_align, label_version=label_version)
        gammas.append(avg_gamma)
        files.append(PurePath(f).name)

def gamma_report(folder):
    ''' Print statistics of gammas for all the per-batch subfolders. '''
    gammas, files, lang = [], [], []
    paths = sorted(Path(folder).iterdir(), key=os.path.getmtime, reverse=True)
    for l in ['es', 'en']:
        for subf in paths:
            if subf.is_dir():
                fne = str(subf.name).split('-')
                if 'iaa' in fne and l in fne:
                    repfile = subf/'gamma_report.txt'
                    with open(repfile, 'r') as f:
                        line = f.readline().strip()
                        gamma = float(line.split()[-1])
                        gammas.append(gamma)
                        files.append(subf.name)
                        lang.append(l)
    gstat = pd.Series(gammas).describe()
    with open(Path(folder)/'gammas.txt', 'w') as report:
        report.write(f'Average gamma: {gstat["mean"]:.3f}\n\n')
        report.write('Distribution of gammas over folders:\n')
        dist_str = '; '.join(f'{stat}: {gstat[stat]:.3f}' for stat in gstat.describe().keys())
        report.write(dist_str + '\n\n')
        curr_lang = None
        for g, f, l in zip(gammas, files, lang):
            if l != curr_lang:
                curr_lang = l
                report.write(f'\nLanguage: {l.upper()}'+'\n\n')
            report.write(f'{f:30} {g:.3f}'+'\n')

def subfolders_as_list(folder):
    ''' Return a list of subfolders of a folder. '''
    return sorted([str(f) for f in Path(folder).iterdir() if f.is_dir()])

if __name__ == '__main__':
    pass
