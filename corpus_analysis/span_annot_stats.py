
from data_tools.seqlab_data.labstudio_parser import ls_parse_all_subfolders
from data_tools.seqlab_data.seqlab_utils import group_by_perm
from settings import *


def load_spans(lang='en'):
    if lang == 'en': dfolder = SEQLABEL_ANNOTS_EN_RAW
    elif lang == 'es': dfolder = SEQLABEL_ANNOTS_ES_RAW
    else: raise ValueError(f'lang {lang} not supported')
    spans = ls_parse_all_subfolders(dfolder)
    spans_by_text = group_by_perm(spans, lambda s: s.text_id)
    print(f'{lang} number of texts: {len(spans_by_text)}')

def raw_span_annot_basic_stats(lang='es'):
    load_spans(lang)

if __name__ == '__main__':
    raw_span_annot_basic_stats()