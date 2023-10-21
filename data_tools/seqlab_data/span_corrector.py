from copy import copy
from typing import List
import regex as re

from networkx import connected_components

from data_tools.seqlab_data.span_graph_utils import create_graph
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlab_utils import group_by_perm
from data_tools.seqlab_data.labstudio_parser import ls_parse_all_subfolders
from data_tools.seqlab_data.span_data_definitions import NONE_LABEL
from settings import AUTOGOLD_ANNOTS_EN_RAW, AUTOGOLD_ANNOTS_ES_RAW

def ends_with_acronym(s):
    # The regex pattern checks for an uppercase letter followed by a dot, repeated at least twice, at the end of the string
    pattern = r'(?:(?:[A-Z]\.){2,})$'
    return bool(re.search(pattern, s))

class SpanCorrector():
    '''
    Performs automatic corrections on a batch of span data.
    '''

    def __init__(self, verbose=False):
        self._verbose = verbose

    def __call__(self, in_spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        spans = self._correct_span_borders(in_spans)
        spans = self._merge_subspans(spans)
        # spans = self._merge_subspans_simple(in_spans)
        return spans

    def _merge_subspans_simple(self, in_spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        '''
        Remove double annotations/spans of the same text - ie, one span inside another with the same label.
        Do this for each author's annotations separately.
        '''
        # group spans by text_id and label
        spans_by_text_id = group_by_perm(in_spans, lambda span: (span.author, span.text_id, span.label))
        # for each group, sort by start, and remove spans completely contained in other spans
        result = []
        for _, s in spans_by_text_id.items():
            spans = sorted(s, key=lambda span: span.start)
            txt_id, txt, auth = spans[0].text_id, spans[0].text, spans[0].author
            again = True
            while again:  # check for cases of one span inside another, remove the inner one
                again = False
                for i in range(len(spans) - 1):
                    sp1 = spans[i]
                    sp2 = spans[i + 1]
                    if sp2.start >= sp1.start and sp2.end <= sp1.end: # if sp2 inside sp1
                        if self._verbose:
                            print(f'SPAN ERROR CORRECT: txt {txt_id} auth {auth} removing [{txt[sp2.start:sp2.end]}] leaving [{txt[sp1.start:sp1.end]}]')
                        spans.pop(i + 1)
                        again = True; break
                    if sp1.start >= sp2.start and sp1.end <= sp2.end: # if sp1 inside sp2
                        if self._verbose:
                            print(f'SPAN ERROR CORRECT: txt {txt_id} auth {auth} removing [{txt[sp1.start:sp1.end]}] leaving [{txt[sp2.start:sp2.end]}]')
                        spans.pop(i)
                        again = True; break
            result.extend(spans)
        return result

    def _merge_subspans(self, in_spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        '''
        Merge fragmented annotations/spans (with the same author and label) - such as one span inside another,
        or overlapping spans - into a single span.
        '''
        # group spans by text_id and label
        spans_by_text_id = group_by_perm(in_spans, lambda span: (span.author, span.text_id, span.label))
        # for each group, sort by start, and remove spans completely contained in other spans
        result = []
        for _, s in spans_by_text_id.items():
            if s[0].label == NONE_LABEL:  # empty annotation(s)
                result.append(s[0])
                continue
            spans = sorted(s, key=lambda span: span.start)
            txt_id, txt, auth, label, txt_num = spans[0].text_id, spans[0].text, spans[0].author, spans[0].label, spans[0].text_num
            edges = []
            for i, sp1 in enumerate(spans):
                for j in range(i, len(spans)): # start at i, so each span will be matched to itself
                    sp2 = spans[j]
                    if sp1.end + 1 <= sp2.start: continue # if sp2 starts after sp1 ends, no overlap
                    # if spans overlap or there is no chars in between:
                    if sp1.end + 1 > sp2.start:
                        edges.append(((i, sp1), (j, sp2)))
            g = create_graph(edges)
            ccs = connected_components(g)
            for cc in ccs:
                c = list(cc)
                # if size of the component is one, then the orig. span will be re-created
                if len(c) == 1: orig_span = c[0][1]
                block_start = min([span.start for _, span in c])
                block_end = max([span.end for _, span in c])
                # initialize new SpanAnnotation, with explicit constructor parameters
                merged_span = SpanAnnotation(text_id=txt_id, text=txt, author=auth, label=label, start=block_start, end=block_end)
                result.append(merged_span)
                if len(c) > 1 and self._verbose:
                    print(f'Overlapping spans detected for text {txt_id}, txt_num {txt_num}, author {auth}:')
                    fragments = [s for _, s in c]
                    # print fragments sorted by start:
                    for s in sorted(fragments, key=lambda span: span.start):
                        print(f'\t|{txt[s.start:s.end+1]}|, start:{s.start}, end: {s.end}, label {s.label}')
                    # print merged fragment:
                    print(f'\tMerged fragment: |{txt[block_start:block_end+1]}|, start:{block_start}, end: {block_end}, label {merged_span.label}')
                else:
                    if merged_span != orig_span and self._verbose:
                        print(f'error:\nmerg. span |{merged_span}|\norig. span |{orig_span}|')
        return result

    def _correct_span_borders(self, spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        '''
        Perform the following corrections, by adjusting the start and end positions of the spans.
        For each span, do:
        While either start and end characters is not alphanumeric
            Remove trailing and leading spaces from spans
            Remove non-alphanumeric characters from the start and end of spans
        While either start - 1 or end +1 characters are alphanumeric:
            Expand span to include the previous and next characters
        :return:
        '''
        result = []
        corrected2orig = {}
        for sp in spans:
            if sp.label == NONE_LABEL: # do not correct 'empty' spans
                result.append(sp)
                continue
            span = copy(sp)
            txt = span.text
            # contract span by removing trailing and leading characters if they're not in the set of allowed characters
            span_left_bord = re.compile(r'[a-zA-Z0-9]|[\u00C0-\u00FF]|[\(\[’"“\']')
            span_right_bord = re.compile(r'[a-zA-Z0-9]|[\u00C0-\u00FF]|[\)\]’"”\']')
            while not span_left_bord.match(txt[span.start]) and span.start < span.end: span.start += 1
            abbrevs = ['inc.', 'ltd.', 'co.', 'corp.', 'jr.', 'wash.', 'ee.', 'uu.', 'bill g.', 'etc.']; stop = False
            while not span_right_bord.match(txt[span.end]) and span.start < span.end:
                setxt = txt[:span.end+1]
                if ends_with_acronym(setxt): break
                for abr in abbrevs:
                    if setxt.lower().endswith(abr): stop = True
                if stop: break
                span.end -= 1
            # expand span to the left and right, but only for alphanumeric characters
            while span.start > 0 and txt[span.start - 1].isalnum(): span.start -= 1
            while span.end < len(txt) - 1 and txt[span.end + 1].isalnum(): span.end += 1
            if span.start < 0 or span.end >= len(txt): # sanity check
                print(f'error: start {span.start} end {span.end} len {len(txt)} id {span.text_id}')
            # if original span has changed, print the correction done
            if self._verbose and (span.start != sp.start or span.end != sp.end):
                print(f'SPAN BORDER CORRECT: txt_id {span.text_id} txt_num {span.text_num} auth {span.author} label {span.label}')
                print(f'  original: {txt[max(0, sp.start - 10):sp.start]}|{txt[sp.start:sp.end + 1]}|{txt[sp.end + 1:min(len(txt), sp.end + 10)]}')
                print(f' corrected: {txt[max(0, span.start - 10):span.start]}|{txt[span.start:span.end+1]}|{txt[span.end+1:min(len(txt), span.end + 10)]}')
            result.append(span)
            span_loc = f'{str(span)}[{span.start}:{span.end}]_{span.text_id}_{span.author}'
            if span_loc not in corrected2orig: corrected2orig[span_loc] = []
            corrected2orig[span_loc].append(sp)
        # print all corrected that have 2 or more original spans
        if self._verbose:
            num_conflations, num_conflated = 0, 0
            for corr in corrected2orig:
                #if len(corrected2orig[corr]) > 1:
                if len(set([str(s) for s in corrected2orig[corr]])) > 1:
                    num_conflations += 1
                    num_conflated += len(corrected2orig[corr])
                    print(f'Corrected span {corr} has {len(corrected2orig[corr])} original spans:')
                    print(';'.join([f'{str(sp)}.{sp.label[0]}' for sp in corrected2orig[corr]]))
                    print()
            print(f'Conflations: {num_conflations} , conflated: {num_conflated}')
        return result

def test_span_correction(lang='en'):
    if lang == 'en': dfolder = AUTOGOLD_ANNOTS_EN_RAW
    else: dfolder = AUTOGOLD_ANNOTS_ES_RAW
    spans = ls_parse_all_subfolders(dfolder)
    SpanCorrector(verbose=True)(spans)

if __name__ == '__main__':
    #test_span_correction('en')
    test_span_correction('es')
