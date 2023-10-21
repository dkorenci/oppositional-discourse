from copy import deepcopy, copy
from typing import List

import pandas as pd

from data_tools.classif_data.data_loaders import corpus_telegram_en_ph1, corpus_telegram_es_ph1, CLASSIF_DF_ID_COLUMN
from data_tools.classif_data.data_utils import add_class_labels, binary_dataset, LBL_CRITIC, LBL_CONSP, CLS_BINARY_IX
from data_tools.raw_dataset_loaders import load_experiment_datasets
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlab_utils import group_by_perm
from data_tools.seqlab_data.span_data_loader import load_annotations
from data_tools.seqlab_data.span_data_definitions import NONE_LABEL, LABEL_DEF

CONSPI_CRITIC = 'CONSPI-OR-CRITIC'

VALUE = 'value'

SPAN_PROPERTY = 'property'

SPAN_CATEGORY = 'span_category'

NUM_WORDS_COLUMN = 'num_words'


def load_common_cdf_annots(lang, verbose=True, gold=False):
    if gold: annots = load_annotations(lang, correct=True, gold=True, annotator_data=False)
    else: annots = load_annotations(lang, correct=True, gold=False)
    if lang == 'en':
        cdf = corpus_telegram_en_ph1()
    elif lang == 'es':
        cdf = corpus_telegram_es_ph1()
    else: raise ValueError(f'Unknown language: {lang}')
    print(f'DF length with 0-0 texts: {len(cdf)}')
    add_class_labels(cdf)
    cdf = binary_dataset(cdf, [LBL_CRITIC, LBL_CONSP])
    print(f'DF length without 0-0 texts: {len(cdf)}')
    # select only rows and annotations that have matching ids
    cdf_ids = set(cdf[CLASSIF_DF_ID_COLUMN].values)
    annot_ids = set(ann.text_id for ann in annots)
    common_ids = cdf_ids.intersection(annot_ids)
    if verbose:
        print(f'# classif. df texts: {len(cdf_ids)}')
        print(f'      # annot texts: {len(annot_ids)}')
        print(f'     # common texts: {len(common_ids)}')
    cdf = cdf[cdf[CLASSIF_DF_ID_COLUMN].isin(common_ids)]
    annots = [ann for ann in annots if ann.text_id in common_ids]
    return cdf, annots, common_ids


class AnnotFeatExtractor:

    def __init__(self, label_def=LABEL_DEF, none_label=NONE_LABEL):
        '''
        :param label_def: map: label name -> label abbreviation
        :param none_label: value of the label that represents no labels
        '''
        self._labdef = label_def
        self._none = none_label

    @property
    def label_def(self): return self._labdef

    def __call__(self, annots: List[SpanAnnotation]):
        self._annots = annots
        self._validate_annots()
        self._process_annots()
        self._calc_flat_feats()
        return deepcopy(self._feats)

    def _validate_annots(self):
        ''' Validates the assumptions for the subsequent processing. '''
        assert len(self._annots) > 0, 'No annotations provided.'
        assert len(set(ann.text_id for ann in self._annots)) == 1, 'Annotations have different text ids.'
        assert len(set(ann.author for ann in self._annots)) == 1, 'Annotations have different text ids.'
        # assert all annotations have values witin the label abbreviations or none
        assert all(ann.label in self._labdef.values() or ann.label == self._none for ann in self._annots), \
            'Annotations have labels that are not in the label\'s definition.'

    def _process_annots(self):
        ''' Calculate basic statistics on the annotations. '''
        self._txt = self._annots[0].text
        self._chars = len(self._txt)
        self._lblcnt = {l: 0 for l in self._labdef.values()}
        self._lblchars = {l: 0 for l in self._labdef.values()}
        for ann in self._annots:
            if ann.label != self._none:
                self._lblcnt[ann.label] += 1
                self._lblchars[ann.label] += ann.end - ann.start
        self._NL = sum(self._lblcnt[l] for l in self._labdef.values()) # number of not-none labels

    def get_feature_sets(self):
        num = [f'{l}_num' for l in self._labdef.values()]
        size_perc = [f'{l}_size_perc' for l in self._labdef.values()]
        size_avg = [f'{l}_size_avg' for l in self._labdef.values()]
        num_perc = [f'{l}_num_perc' for l in self._labdef.values()]
        return num, size_perc, size_avg, num_perc

    CNT_LABEL = 'CNT'
    SIZE_PERC_LABEL = 'PERC_TEXT'
    SIZE_PERC_AVG_LABEL = 'PERC_TEXT_AVG'
    CNT_PERC_LABEL = 'PROPORTION'
    NUM_LABELS = 'NUM_LABELS'

    def _calc_flat_feats(self):
        '''
        Calc. features in the form (label, property): value
        '''
        f = {(l, self.CNT_LABEL): self._lblcnt[l] for l in self._labdef.values()}
        for l in self._labdef.values():
            f[(l, self.SIZE_PERC_LABEL)] = self._lblchars[l] / self._chars * 100
            f[(l, self.SIZE_PERC_AVG_LABEL)] = f[(l, self.SIZE_PERC_LABEL)] / self._lblcnt[l] if self._lblcnt[l] > 0 else 0
        if self._NL > 0:
            for l in self._labdef.values():
                f[(l, self.CNT_PERC_LABEL)] = self._lblcnt[l] / self._NL
        f[self.NUM_LABELS] = self._NL
        self._feats = f

def create_seaborn_boxplot_df(annots, cdf, fextr, text_level_only=False):
    '''
    Dataframe for seaborn parallel boxplots plot.
    Rows are textauthor-label-labelproperty combinations,
    columns are text features, and 'value' with numeric values.
    '''
    ann_byauthtxt = group_by_perm(annots, lambda a: (a.text_id, a.author))
    cdf.set_index(CLASSIF_DF_ID_COLUMN, inplace=True, verify_integrity=True)
    rows = []
    for authtxt, annots in ann_byauthtxt.items():
        f = fextr(annots)
        txt_id = authtxt[0]
        text = annots[0].text; num_words = len(text.split())
        row = {'author': authtxt[1], 'text_id': txt_id, NUM_WORDS_COLUMN: num_words,
               CONSPI_CRITIC: 'critical' if cdf.loc[txt_id, CLS_BINARY_IX] == 0 else 'conspiracy'}
        if text_level_only:
            rows.append(row)
            continue
        for prop, val in f.items():
            rcpy = copy(row)
            if isinstance(prop, tuple):
                sl, lab = prop
                rcpy[SPAN_CATEGORY] = sl
                rcpy[SPAN_PROPERTY] = lab
                rcpy[VALUE] = val
            rows.append(rcpy)
    # dataframe from the features
    df = pd.DataFrame(rows)
    return df

def create_export_df(annots, cdf, fextr):
    '''
    Text-author per row, colums are basic data, plus the counts of each label's occurrence in the annotations.
    '''
    ann_byauthtxt = group_by_perm(annots, lambda a: (a.text_id, a.author))
    cdf.set_index(CLASSIF_DF_ID_COLUMN, inplace=True, verify_integrity=True)
    rows = []
    for authtxt, annots in ann_byauthtxt.items():
        f = fextr(annots)
        txt_id = authtxt[0]
        text = annots[0].text; num_words = len(text.split())
        row = {'annotator': authtxt[1], 'text_id': txt_id, 'num_words': num_words, 'text': text,
               CONSPI_CRITIC: 'critical' if cdf.loc[txt_id, CLS_BINARY_IX] == 0 else 'conspiracy'}
        for span_label in fextr.label_def.values():
            row[span_label] = f[(span_label, fextr.CNT_LABEL)]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def get_df(lang, verbose=True, text_level_only=False):
    ''' Load the df with the statistics. '''
    cdf, annots, common_ids = load_common_cdf_annots(lang, verbose=verbose)
    fextr = AnnotFeatExtractor()
    return create_seaborn_boxplot_df(annots, cdf, fextr, text_level_only=text_level_only)

def create_spanfeatures_dataset(lang, verbose=True, gold=False):
    #cdf, annots, common_ids = load_common_cdf_annots(lang, verbose=verbose, gold=gold)
    cdf, annots = load_experiment_datasets(lang, verbose=verbose)
    fextr = AnnotFeatExtractor()
    return create_export_df(annots, cdf, fextr)

if __name__ == '__main__':
    get_df('es')
    #count_for_categs()
    #txtperc_for_categs()



