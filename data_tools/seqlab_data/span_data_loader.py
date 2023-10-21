import copy
from typing import List

from data_tools.seqlab_data.auto_gold_labels import create_gold_lables
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.labstudio_parser import ls_parse_all_subfolders
from data_tools.seqlab_data.seqlab_utils import process_duplicate_span_labels
from data_tools.seqlab_data.span_corrector import SpanCorrector
from settings import ANNOTS_EN_RAW, ANNOTS_EN_RAW_DEV, ANNOTS_ES_RAW, ANNOTS_ES_RAW_DEV


def load_annotations(lang, correct=True, gold=True, annotator_data=False, id_filter=None, dev=False) -> List[SpanAnnotation]:
    '''
    Load span data from .json Label studio files, and optionally apply corrections and create gold labels.
    :param correct: if true, apply corrections to the spans
    :param gold: if true, create and convert gold labels for the spans
    :param annotator_data: if true, keep the original annotator spans along with the gold labels
    :return:
    '''
    # load spans and apply trasnformations
    spans = load_raw_annotation_dataset(lang, dev=dev)
    if id_filter:
        id_filter = set(id_filter)
        spans = [span for span in spans if span.text_id in id_filter]
    return process_annotations(spans, correct=correct, gold=gold, annotator_data=annotator_data)

def process_annotations(spans: List[SpanAnnotation], correct=True, gold=True,
                        annotator_data=False, verbose=False) -> List[SpanAnnotation]:
    if correct: spans = SpanCorrector(verbose=verbose)(spans)
    spans = process_duplicate_span_labels(spans, verbose=verbose)
    if gold:
        if annotator_data: orig_spans = copy.deepcopy(spans)
        spans = create_gold_lables(spans)
        if annotator_data: spans.extend(orig_spans)
    return spans

def load_raw_annotation_dataset(lang, dev=False) -> List[SpanAnnotation]:
    '''
    Load annotations, with applied gold corrections.
    '''
    if lang == 'en': dset_folder = ANNOTS_EN_RAW if not dev else ANNOTS_EN_RAW_DEV
    elif lang == 'es': dset_folder = ANNOTS_ES_RAW if not dev else ANNOTS_ES_RAW_DEV
    else: raise ValueError(f'Unknown language {lang}')
    return ls_parse_all_subfolders(dset_folder)
