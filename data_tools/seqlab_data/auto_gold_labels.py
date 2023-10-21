'''
Functionalit for automatically generating gold labels for a sequence labeling task,
from the multi-author human gold labels.
'''
import copy
from pathlib import Path
import random
from typing import List, Callable
from matplotlib import pyplot as plt
from pyannote.core import Segment
from pygamma_agreement import Continuum
from pygamma_agreement.notebook import Notebook

from gamma_iaa.analyze_gamma import print_annots_cont, set_labels, SHORT_LABELS_V2
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlab_utils import group_by_perm
from data_tools.seqlab_data.labstudio_parser import ls_parse_all_subfolders
from data_tools.seqlab_data.span_data_definitions import NONE_LABEL
from data_tools.seqlab_data.span_corrector import SpanCorrector
from data_tools.seqlab_data.span_graph_utils import create_graph, connected_components
from settings import *

GOLD_LABEL_AUTHOR = 'gold_label'


def spans_overlap(span1, span2):
    return len(set(range(span1.start, span1.end+1)).intersection(set(range(span2.start, span2.end+1)))) > 0

def good_span_match(span1: SpanAnnotation, span2: SpanAnnotation):
    ''' True if the spans have the same label, and if the overlap is at least 85% of both pans. '''
    if span1.label != span2.label: return False
    overlap = len(set(range(span1.start, span1.end)).intersection(set(range(span2.start, span2.end))))
    return overlap >= 0.85 * len(span1) and overlap >= 0.85 * len(span2)

def span_match(span1: SpanAnnotation, span2: SpanAnnotation):
    ''' True if the spans have the same label, and the overlap is not empty. '''
    if span1.label != span2.label: return False
    return spans_overlap(span1, span2)

def match_spans(spans1 : List[SpanAnnotation], spans2 : List[SpanAnnotation], match_crierion:Callable = spans_overlap):
    '''
    Match spans from two lists of spans, based on overlap.
    :return: Pairs of matching indices, and a dict: spans1 index -> spans2 indices, and the reverse dict
    '''
    pairs = []
    edges12 = {i: [] for i in range(len(spans1))}
    edges21 = {i: [] for i in range(len(spans2))}
    for i, span1 in enumerate(spans1):
        for j, span2 in enumerate(spans2):
            if match_crierion(span1, span2):
                pairs.append((i, j))
                edges12[i].append(j)
                edges21[j].append(i)
    return pairs, edges12, edges21

def perform_merge(merges: List[List[SpanAnnotation]], check=True):
    ''' Merge together all the span lists in the list of merges.
      Each span list must have the same label, and the resulting span will extend from the leftmost
      start to the rightomst end of spans, and have the same label. '''
    merged_spans = []
    for spans in merges:
        if len(spans) == 0: continue
        label = spans[0].label
        text_id = spans[0].text_id
        text = spans[0].text
        # check all spans have the same text_id, label, and text
        text_ids = set([span.text_id for span in spans])
        if check:
            if len(text_ids) != 1:
                raise ValueError(f'Found more than one text_id in the list of spans, '
                                 f'or no spans at all (empty list). IDs: {text_ids}')
            labels = set([span.label for span in spans])
            if len(labels) != 1:
                raise ValueError(f'Found more than one label in the list of spans, '
                                 f'or no spans at all (empty list). Labels: {labels}')
            texts = set([span.text for span in spans])
            if len(texts) != 1:
                raise ValueError(f'Found more than one text in the list of spans, '
                                 f'or no spans at all (empty list). Texts: {texts}')
        author = GOLD_LABEL_AUTHOR
        start = min([span.start for span in spans])
        end = max([span.end for span in spans])
        merged_spans.append(SpanAnnotation(text_id=text_id, text=text, label=label, author=author, start=start, end=end))
    return merged_spans

def gold_align_spans(spans: List[SpanAnnotation], separate_goodmatch=False, verbose=False) -> List[SpanAnnotation]:
    ''' Given a list of spans for a single file, merge them into
    the list of gold spans according to a set of rules.'''
    # check all spans have the same text_id
    text_ids = set([span.text_id for span in spans])
    if len(text_ids) != 1:
        raise ValueError(f'Found more than one text_id in the list of spans, '
                         f'or no spans at all (empty list). IDs: {text_ids}')
    txt_id = text_ids.pop()
    # check and extract authors and their respective spans
    auth2spans = group_by_perm(spans, lambda span: span.author)
    authors = list(set(auth2spans.keys()))
    if len(authors) != 2:
        raise ValueError(f'Number of authors in the list of spans must be two. Text id: {txt_id}, Authors: {authors}')
    auth1, auth2 = authors
    spans1, spans2 = auth2spans[auth1], auth2spans[auth2]
    # tackle 'empty' annotations case
    if (len(spans1) == 1 and spans1[0].label == NONE_LABEL) or \
        (len(spans2) == 1 and spans2[0].label == NONE_LABEL): #empty annotation for one of the authors
            return [SpanAnnotation(text_id=txt_id, text=spans1[0].text, label=NONE_LABEL, author=GOLD_LABEL_AUTHOR)]
    # find, record, and remove good matches
    if separate_goodmatch:
        pairs, edges12, edges21 = match_spans(spans1, spans2, good_span_match)
        # check that every span1 is matched to at most one span2, and vice versa
        for i, span1 in enumerate(spans1):
            if len(edges12[i]) > 1:
                raise ValueError(f'Span {i} of author {auth1} is good-matched to more than one span of author {auth2}.')
        for j, span2 in enumerate(spans2):
            if len(edges21[j]) > 1:
                raise ValueError(f'Span {j} of author {auth2} is good-matched to more than one span of author {auth1}.')
        gold_merges = [[span1[i], span2[j]] for i, j in pairs]
        # remove spans from pairs from the lists of spans
        matched1 = set([pair[0] for pair in pairs])
        matched2 = set([pair[1] for pair in pairs])
        spans1 = [span for i, span in enumerate(spans1) if i not in matched1]
        spans2 = [span for i, span in enumerate(spans2) if i not in matched2]
    else: gold_merges = []
    # perform matching of the remaining spans
    pairs, edges12, edges21 = match_spans(spans1, spans2, span_match)
    bp_pairs = [ ((i, spans1[i]), (j, spans2[j])) for i, j in pairs ]
    g = create_graph(bp_pairs)
    ccs = connected_components(g)
    merges = []
    for c in ccs:
        if len(c) > 2 and verbose:
            print('Chain detected:')
            print('Text:', list(c)[0][1].text)
            for _, span in c:
                print(f'{span.author}, {span.start}:{span.end}, {span.label} , text: {span.text[span.start:span.end]}')
            print()
            # raise ValueError(f'Found a chain of spans with length > 2: {c}')
        if len(c) >= 2:
            merges.append([span for _, span in c])
    merges.extend(gold_merges)
    merges = perform_merge(merges)
    if len(merges) == 0: # empty gold, add a single NONE_LABEL span
        if verbose:
            print(f'Empty gold for text {spans[0].text_id}, adding a single NONE_LABEL span.')
        merges = [SpanAnnotation(text_id=spans[0].text_id, text=spans[0].text, label=NONE_LABEL, author=GOLD_LABEL_AUTHOR)]
    assert len(merges) > 0, f'Empty gold for text {spans[0].text_id}.'
    return merges

def span_to_segment(span: SpanAnnotation, all_spans) -> Segment:
    '''
    Convert SpanAnnotation to Segment for the pygamma format.
    Special case is the NONE_LABEL span, which has to have start and end calculated from the other spans,
    for the purposes of gamma computation, and visualization.
    '''
    if span.label != NONE_LABEL:
        return Segment(start=span.start, end=span.end)
    else:
        empty_len, empty_dist = 5, 10
        if len(all_spans) > 0 and set([span.end for span in all_spans]) != set([None]):
            max_pos = max([span.end for span in all_spans if span.end is not None])
            max_pos += empty_dist
        else:
            max_pos = 1
        return Segment(start=max_pos, end=max_pos+empty_len)

def visualize_gold_vs_orig(spans, gold_spans, sample_size=10, out_folder='.', rseed=29413):
    # extract text_ids from spans, and gold spans, separately
    text_ids = set([span.text_id for span in spans])
    gold_text_ids = set([span.text_id for span in gold_spans])
    # assert equality
    assert text_ids == gold_text_ids
    # take a random sample of text ids
    text_ids = sorted(list(text_ids)) # sort predictable sampling depending on rseed only
    random.seed(rseed)
    sample_ids = random.sample(text_ids, min(sample_size, len(text_ids)))
    # group spans, and gold spans, by text id
    id2spans = group_by_perm(spans, lambda span: span.text_id)
    id2gold_spans = group_by_perm(gold_spans, lambda span: span.text_id)
    # graphics settings
    w, h, my_dpi = 1928, 1272, 300
    notebook = Notebook()
    for text_id in sample_ids:
        # get spans with the text_id, and gold spans with the text_id, separately
        spans4id = id2spans[text_id]
        text = spans4id[0].text
        if text_id not in gold_text_ids:
            gold_spans4id = [SpanAnnotation(text_id=text_id, text=text, label=NONE_LABEL, author=GOLD_LABEL_AUTHOR)]
        else:
            gold_spans4id = id2gold_spans[text_id]
        continuum = Continuum()
        all_spans = spans4id + gold_spans4id
        for span in all_spans:
            continuum.add(span.author, span_to_segment(span, all_spans), span.label)
        gamma_results = continuum.compute_gamma(n_samples=10)
        # print all annotations to a text file
        set_labels(2)  # for the print_annots_cont function to work correctly
        # convert spans to the format required by the print_annots_cont function
        SHORT_LABELS_REV = {v: k for k, v in SHORT_LABELS_V2.items()}; SHORT_LABELS_REV[NONE_LABEL] = NONE_LABEL
        span2annot = lambda span: { 'text_id': span.text_id, 'text': span.text, 'label': SHORT_LABELS_REV[span.label],
                                    'author': span.author, 'text_num': span.text_num, 'start': span.start, 'end': span.end }
        annots = [ span2annot(span) for span in all_spans ]
        with open(Path(out_folder)/f'{text_id}.txt', 'w') as f:
            print_annots_cont(f, annots, continuum)
        # visualize the annotation using pygamma, save as pdf
        fig, ax = plt.subplots(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)  # plt.subplots()
        notebook.plot_alignment_continuum(gamma_results.best_alignment, ax=ax, labelled=True)
        plt.tight_layout()
        outfile = Path(out_folder) / f'{text_id}.pdf'
        plt.savefig(outfile, dpi=my_dpi)

def create_gold_labels_batch(out_folder=None, lang='en', vis_test_sample=False):
    '''
    Create gold labels from the all raw annotations for the specified language.
    Apply span correction beforehand.
    '''
    if lang == 'en': dfolder = AUTOGOLD_ANNOTS_EN_RAW
    else: dfolder = AUTOGOLD_ANNOTS_ES_RAW
    if out_folder is None: out_folder = Path(dfolder).parent / f'{lang}-gold'
    spans = ls_parse_all_subfolders(dfolder)
    orig_spans = copy.deepcopy(spans)
    spans = SpanCorrector(verbose=True)(spans)
    id2spans = group_by_perm(spans, lambda span: span.text_id)
    gold = []
    for text_id, sp in id2spans.items():
        gold.extend(gold_align_spans(sp))
    if vis_test_sample:
        print('Visualizing sample of gold labels')
        vis_folder = Path(dfolder).parent / f'{lang}-gold-sample-vis'
        vis_folder.mkdir(exist_ok=True)
        visualize_gold_vs_orig(orig_spans, gold, out_folder=vis_folder, sample_size=vis_test_sample)

def create_gold_lables(spans: List[SpanAnnotation], verbose=False) -> List[SpanAnnotation]:
    '''
    Modular version of gold span creation, working directly with the SpanAnnotation objects,
    and not applying any corrections or modifications to them.
    '''
    id2spans = group_by_perm(spans, lambda span: span.text_id)
    gold = []
    for text_id, sp in id2spans.items():
        try:
            gold.extend(gold_align_spans(sp, verbose=verbose))
        except Exception as e:
            print(f'Error in gold_align_spans for text_id={text_id}: {e}')
            raise e
    return gold

if __name__ == '__main__':
    create_gold_labels_batch(None, 'en', vis_test_sample=200)
    #bipartite_demo()
