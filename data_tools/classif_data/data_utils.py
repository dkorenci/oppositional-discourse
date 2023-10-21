# COLUMN NAMES SHORTCUTS
from data_tools.classif_data.data_loaders import corpus_telegram_es_ph1, corpus_telegram_en_ph1, CLASSIF_DF_ID_COLUMN

GOLD_CONSPI = 'GS_CONSPI-THEORY'
GOLD_CRITIC = 'GS_CRITICAL_THINKING'
NUM_WORDS = 'word_count'
ANNOT_CONSPI = [None] + [f'Annotator-{i}_CONSPI-THEORY' for i in range(1,4)]
ANNOT_CRITIC = [None] + [f'Annotator-{i}_CRITICAL_THINKING' for i in range(1,4)]
TXT_COLUMN = 'body'

CLS_BINARY_IX = 'class-binary'
CLS_TRNRY_IX = 'class-ternary'
CLS_TRNRY_LAB = 'class-ternary-lab'
TRNRY_CLASSES = (0, 1, 2)
LBL_NOCONSP, LBL_CRITIC, LBL_CONSP = 'NO_CONSP', 'CRITIC', 'CONSP'
TRNRY_LABELS = {0: LBL_NOCONSP, 1: LBL_CRITIC, 2: LBL_CONSP}

def df_info(df):
    ''' print basic pd.DataFrame info '''
    print('COLUMNS:')
    for c in df: print (c)
    print('DF PREVIEW:')
    print(df)

def add_class_labels(df):
    # ternary 0 - no consp. , 1 - critical, 2 - conspiratorial
    # set both numeric and txt class labels
    for col in [CLS_TRNRY_IX, CLS_TRNRY_LAB]:
        cl0, cl1, cl2 = TRNRY_CLASSES if col == CLS_TRNRY_IX \
                            else tuple(TRNRY_LABELS[c] for c in TRNRY_CLASSES)
        df[col] = cl0
        df.loc[df[GOLD_CRITIC] == 1, col] = cl1
        df.loc[df[GOLD_CONSPI] == 1, col] = cl2
    return df

def binary_dataset(df, classes):
    '''
    create a dataset for binary classification between two classes.
    :param df: dataset, with add_class_labels() applied
    :param classes: a list of two class labels from LBL_NOCONSP, LBL_CRITIC, LBL_CONSP
    :return:
    '''
    assert(len(classes) == 2)
    assert(set(classes).issubset(set(TRNRY_LABELS.values())))
    cl0, cl1 = classes[0], classes[1]
    df = df[(df[CLS_TRNRY_LAB] == cl0) | (df[CLS_TRNRY_LAB] == cl1)].copy()
    bincls = {cl0: 0, cl1: 1}
    df[CLS_BINARY_IX] = df[CLS_TRNRY_LAB].apply(lambda c: bincls[c])
    df.reset_index(drop=True, inplace=True)
    return df

def create_classif_dataset(lang, classes='all', output='dataframe'):
    '''
    Create a dataset to be used for classification
    :param classes: 'all' for ternary or a list of two class labels
    :param output: 'dataframe' - orig. dataframe with new columns, 'sklean' - X, label pair
    :return:
    '''
    if lang.lower() == 'es': df = corpus_telegram_es_ph1()
    elif lang.lower() == 'en': df = corpus_telegram_en_ph1()
    add_class_labels(df)
    if classes != 'all': df = binary_dataset(df, classes)
    if output == 'dataframe': return df
    elif output == 'sklearn':
        if classes == 'all': target = CLS_TRNRY_IX
        else: target = CLS_BINARY_IX
        texts, classes, ids = df[TXT_COLUMN], df[target], df[CLASSIF_DF_ID_COLUMN]
        return texts, classes, ids

if __name__ == '__main__':
    create_classif_dataset('en', output='sklearn')


