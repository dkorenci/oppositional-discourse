from settings import *

import pandas as pd

CLASSIF_DF_ID_COLUMN = 'NUM'

def classif_corpus_raw_df(lang):
    ''' Load 'raw', ie, annotation-phase output, table as dataframe. '''
    if lang == 'es': fpath = CORPUS_TELEGRAM_ES_PHASE1
    elif lang == 'en': fpath = CORPUS_TELEGRAM_EN_PHASE1
    else: raise ValueError(f'Unknown language {lang}')
    return pd.read_excel(fpath, dtype={'author': str})

def corpus_telegram_es_ph1():
    df = classif_corpus_raw_df('es')
    return df

def corpus_telegram_en_ph1():
    df = classif_corpus_raw_df('en')
    return df

if __name__ == '__main__':
    corpus_telegram_en_ph1()