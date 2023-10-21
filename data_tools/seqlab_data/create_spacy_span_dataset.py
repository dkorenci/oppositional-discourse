'''
Create convert texts to spacy Doc objects with annotation and category data.
'''
import copy
from typing import List, Tuple
from spacy import Language

from data_tools.seqlab_data.span_data_loader import load_annotations, load_raw_annotation_dataset
from data_tools.spacy_utils import *
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlab_utils import group_by_perm, \
    match_span_annots_to_spacy_spans, process_duplicate_span_labels
from data_tools.seqlab_data.url_tokenization import SpacyURLTokenizer


def create_annot_span_tuple(span_annot: SpanAnnotation, match: Span) -> Tuple[str, int, int, str]:
    '''
    Create a tuple of (label, start, end, author) from a span annotation and a spacy span.
    '''
    return span_annot.label, match.start, match.end, span_annot.author

def get_annoation_tuples_from_doc(doc:Doc):
    '''
    Helper to get annotation tuples from a spacy doc - just reads off the extension property
    that holds the list of annotations, and makes a copy.
    '''
    return list(doc._.get(ON_DOC_EXTENSION))

def get_doc_id(doc:Doc):
    return doc._.get(ON_DOC_ID)

def get_doc_class(doc:Doc):
    return doc._.get(ON_DOC_CLS_EXTENSION)

def add_span_annotation_to_doc(doc:Doc, span_annot:SpanAnnotation, match:Span, format:str='tuple'):
    '''
    Add spann annotation data to the spacy doc: author, label, token indices.
    :param format: if 'annot' store as SpanAnnotation, otherwise as a tuple of (label, start, end, author)
    :return:
    '''
    if format == 'annot':
        annot = copy.copy(span_annot)
        annot.spacy_start = match.start
        annot.spacy_end = match.end
    elif format == 'tuple':
        annot = create_annot_span_tuple(span_annot, match)
    else: raise ValueError(f'Unknown format {format}')
    doc._.get(ON_DOC_EXTENSION).append(annot)

def create_spacy_doc_from_annots(nlp: Language, spans: List[SpanAnnotation], verbose=False, debug=False):
    '''
    Create a spacy docs form spans with the same text id.
    Spans are converted to spacy Spans, and added to the doc,
    and span borders are properly aligned with token borders.
    '''
    # check that all spans have the same text id
    text_ids = set([span.text_id for span in spans])
    if len(text_ids) != 1: raise ValueError(f'All spans must have the same text id, the id set is: {text_ids}')
    text_id = text_ids.pop()
    text = spans[0].text
    # create spacy doc
    try:
        doc = nlp(text)
    except ValueError as e:
        print(f'Error while spacy-processing text id {text_id}\ntext:[{text}]\n\n{e}')
        raise e
    if debug: print('SPANS: ', ';'.join([f'{str(span)}.{span.label}@{span.start}' for span in spans]))
    span_matching = match_span_annots_to_spacy_spans(doc, spans, debug=debug)
    unmatched = []
    text_printed = False
    doc._.set(ON_DOC_ID, text_id)
    #if verbose: print('processing text: ', text_id)
    for span_annot in spans:
        if span_annot in span_matching:
            add_span_annotation_to_doc(doc, span_annot, span_matching[span_annot], format='tuple')
            if debug:
                print(f'matched span txt: [{str(span_annot)}.{span_annot.label}]')
                text_printed = True
        else:
            if verbose:
                if not unmatched: # first unmatched
                    print(f'Unmatched spans exist for text id {text_id}:')
                    print('doc tokens: ', ';'.join([token.text for token in doc]))
                    print(f'orig. text: [{text}]')
                print(f'unmatched span txt: [{str(span_annot)}.{span_annot.label}]')
                text_printed = True
            unmatched.append(span_annot)
    # if verbose:
    #     #print(f'matched {len(spans) - len(unmatched)} spans, unmatched {len(unmatched)} spans')
    #     if random.uniform(0, 1) < 0.01:
    #         # print text of span_annots and matching spacy spans
    #         print('matched spans:')
    #         for span_annot in spans:
    #             if span_annot in span_matching:
    #                 match = span_matching[span_annot]
    #                 print(f'[{span_annot.text}] -> [{match.text}]')
    #         if not unmatched: print
    if text_printed and (verbose or debug): print()
    return doc, unmatched

def span_annotations_to_spacy_docs(annots: List[SpanAnnotation], lang: str, debug=False,
                                   verbose=False, stop_after=None) -> List[Doc]:
    spans_by_text_id = group_by_perm(annots, lambda span: span.text_id)
    docs = []
    num_unmatched = 0
    nlp = create_spacy_model(lang, fast=True)
    nlp.tokenizer = SpacyURLTokenizer(nlp)
    num_texts = 0; num_spans = 0; num_texts_with_unmatched = 0
    for text_id, annots in spans_by_text_id.items():
        num_spans += len(annots)
        doc, unmatched = create_spacy_doc_from_annots(nlp=nlp, spans=annots, verbose=verbose, debug=debug)
        docs.append(doc)
        num_unmatched += len(unmatched)
        num_texts += 1; num_texts_with_unmatched += 1 if len(unmatched) > 0 else 0
        if stop_after and num_texts >= stop_after: break
        if num_texts % 200 == 0: print(f'processed {num_texts} texts')
    perc_span = num_unmatched/num_spans if num_spans > 0 else 0
    print(f'Unmatched spans: {num_unmatched}/{num_spans}, percentage {perc_span*100:.3f}')
    perc_txt = num_texts_with_unmatched/num_texts if num_texts > 0 else 0
    print(f'Texts with unmatched span {num_texts_with_unmatched}/{num_texts}, percentage {perc_txt*100:.3f}')
    return docs

def check_spans(spans: List[SpanAnnotation], tag, search_str):
    for sp in spans:
        if str(sp) == search_str:
            print(f'{tag}:', sp.text_id, sp.start, sp.end, sp.label, sp.author)

def convert_spans_to_spacy(lang, correct=True, gold=True, annotator_data=False, id_filter=None,
                           debug=False, verbose=False, stop_after=None):
    ''' Load all span data and convert to spacy Docs. '''
    print('Loading spans...')
    spans = load_annotations(lang, correct, gold, annotator_data, id_filter=id_filter)
    if id_filter:
        id_filter = set(id_filter)
        spans = [span for span in spans if span.text_id in id_filter]
    spans = sorted(spans, key=lambda span: span.text_id)
    define_spacy_extensions()
    print('Converting spans to spacy docs...')
    return span_annotations_to_spacy_docs(spans, lang, debug=debug, verbose=verbose, stop_after=stop_after)

def analyze_duplicate_labels(lang):
    spans = load_raw_annotation_dataset(lang)
    process_duplicate_span_labels(spans, output_labelsets=True, stats=True)

if __name__ == '__main__':
    #load_spans('en', id_filter=['1321650946_15844'])
    #convert_spans_to_spacy('en', verbose=True, stop_after=None)
    convert_spans_to_spacy('en', id_filter=['1457099754_4573'], debug=True, verbose=True)
    #convert_spans_to_spacy('es')
    #analyze_duplicate_labels('es')
