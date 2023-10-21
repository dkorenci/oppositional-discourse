from typing import Tuple, List
from pandas import DataFrame

from data_tools.classif_data.data_loaders import corpus_telegram_en_ph1, corpus_telegram_es_ph1, CLASSIF_DF_ID_COLUMN, \
    classif_corpus_raw_df
from data_tools.classif_data.data_utils import add_class_labels, binary_dataset, LBL_CRITIC, LBL_CONSP
from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlab_utils import group_by_perm
from data_tools.seqlab_data.span_data_loader import load_annotations, load_raw_annotation_dataset, process_annotations

CONSPI_CRITIC = 'CONSPI-OR-CRITIC'

VALUE = 'value'

SPAN_PROPERTY = 'property'

SPAN_CATEGORY = 'span_category'

NUM_WORDS_COLUMN = 'num_words'

def load_raw_datasets(lang, verbose=True, gold=False):
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

def load_experiment_datasets(lang, verbose=False, gold=True, annot_data=False) -> Tuple[DataFrame, List[SpanAnnotation]]:
    '''
    Final version of the datasets for the experiment: annotation ids matched with classif. ids,
    classif. dataset with conspiracy==0, critical==1 binary labels,
    annotations corrected, with gold labels
    :param lang:
    :param verbose:
    :return:
    '''
    cdf = classif_corpus_raw_df(lang)
    if verbose: print(f'DF length with 0-0 texts: {len(cdf)}')
    add_class_labels(cdf)
    cdf = binary_dataset(cdf, [LBL_CRITIC, LBL_CONSP])
    if verbose: print(f'DF length without 0-0 texts: {len(cdf)}')
    cdf_ids = set(cdf[CLASSIF_DF_ID_COLUMN].values)
    # extract raw annotation data
    annots = load_raw_annotation_dataset(lang)
    annots_ids = set(ann.text_id for ann in annots)
    num_annot_unfiltered = len(annots_ids)
    # extract missing ids, in both directions
    annots_missing_cdf = list(set([ann for ann in annots if ann.text_id not in cdf_ids]))
    annots_missing_spandata = cdf_ids.difference(annots_ids)
    # filter out annotations that are not in the classif. df
    annots = [ann for ann in annots if ann.text_id in cdf_ids]
    num_annots_filtered = len(set(ann.text_id for ann in annots))
    if verbose: print(f'Annotation texts: {num_annot_unfiltered} -> {num_annots_filtered}, diff: {num_annot_unfiltered - num_annots_filtered}')
    annots = process_annotations(annots, correct=True, gold=gold, annotator_data=annot_data)
    final_annots_ids = set(ann.text_id for ann in annots)
    #if len(cdf) != len(group_by_perm(annots, lambda ann: ann.text_id)):
    if cdf_ids != final_annots_ids:
        print(f'WARNING: classif. df and annotations have different number of texts: '
              f'{len(cdf)} vs {len(group_by_perm(annots, lambda ann: ann.text_id))}')
        annots_by_id = group_by_perm(annots_missing_cdf, lambda ann: ann.text_id)
        #for text_id, annots in annots_by_id.items():
        #for text_id in annots_missing_spandata: print(f'{text_id}')
    return cdf, annots

if __name__ == '__main__':
    load_experiment_datasets('en', verbose=True)
    #load_experiment_datasets('es', verbose=True)
