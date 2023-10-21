import numpy as np
import pandas as pd

from data_tools.classif_data.data_loaders import CLASSIF_DF_ID_COLUMN
from data_tools.classif_data.data_utils import TXT_COLUMN
from data_tools.raw_dataset_loaders import load_experiment_datasets
from data_tools.seqlab_data.seqlab_utils import group_by_perm


def create_spancateg_classif_dataset(lang, span_category, verbose=False):
    '''
    Create a binary classification dataset based on a given span category:
    0 == no span with the specified category in the text, 1 == at least one span with the specified category.
    :param span_category: one letter category label
    :return:
    '''
    cdf, span_annots = load_experiment_datasets(lang, verbose=False)
    group_by_id = group_by_perm(span_annots, lambda ann: ann.text_id)
    texts, classes, text_ids = [], [], []
    for txt, txt_id in zip(cdf[TXT_COLUMN], cdf[CLASSIF_DF_ID_COLUMN]):
        if txt_id in group_by_id:
            spans = group_by_id[txt_id]
            if any(span.label == span_category for span in spans): classes.append(1)
            else: classes.append(0)
        else: classes.append(0)
        texts.append(txt)
        text_ids.append(txt_id)
    if verbose:
        print(f'Number of texts: {len(texts)}')
        print(f'Number of classes: {len(classes)}')
        # calculate and print the class distribution
        class0 = sum(1 for c in classes if c == 0)
        class1 = sum(1 for c in classes if c == 1)
        # print number and percentages of classes
        print(f'Class -{span_category}: {class0} ({class0/len(classes)*100:.2f}%)')
        print(f'Class +{span_category}: {class1} ({class1/len(classes)*100:.2f}%)')
    # convert texts to a pandas series of texts, and classes to numpy array, and return
    return pd.Series(texts), np.array(classes), pd.Series(text_ids)

if __name__ == '__main__':
    create_spancateg_classif_dataset('en', 'F', verbose=True)
    create_spancateg_classif_dataset('es', 'F', verbose=True)
