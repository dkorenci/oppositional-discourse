'''
Apply custom list of words indicating political violence to calculate proportion of violent words in a text.
'''
from typing import List, Tuple, Dict

import pandas as pd
import editdistance

from corpus_analysis.violence_extract.llm_violent_words import Lemmatizer
from settings import VIOLENT_WORDS_ES, VIOLENT_WORDS_EN


class ViolenceIndexCalculator():

    def __init__(self, lang):
        self._lang = lang
        self._violent_words = set(vw.lower() for vw in load_violent_words(lang))
        self._categ_map = load_violent_categ_map(lang)
        self._lemmatizer = Lemmatizer(lang)

    def __call__(self, txt: str) -> float:
        '''
        :param txt: text to calculate violence index for
        :return: proportion of violent words in text
        '''
        N, num_violent = self.violent_idx_data(txt)
        if N == 0: return 0
        return num_violent / N

    def violent_idx_data(self, txt: str, categs=False) -> Tuple[int, int]:
        ''' return length of text (num. tokens), and number of violent words, separately '''
        lemmas = self._lemmatizer.lemmatize_text(txt)
        N = len(lemmas)
        if N == 0: return 0, 0
        violent_words = [lemma for lemma in lemmas if lemma in self._violent_words]
        num_violent = len(violent_words)
        if categs:
            categs_present = [self._categ_map[lemma] for lemma in lemmas if lemma in self._violent_words]
            categs_canonical = [find_string_with_edit_distance(categ, VIOLENT_CATEGS) for categ in categs_present]
            categs_canonical = [categ for categ in categs_canonical if categ is not None]
            categs_counts = { categ: categs_canonical.count(categ) for categ in categs_canonical }
            # print each of the following lists, and the map, each in one line, elements concatenated as string, ; separated
            # print(txt, '\n')
            # print(f'violent_words: [{";".join(violent_words)}]')
            # print(f'categs_present: [{";".join(categs_present)}]')
            # print(f'categs_canonical: [{";".join(categs_canonical)}]')
            # print(f'categs_counts: [{";".join([f"{categ}:{count}" for categ, count in categs_counts.items()])}]')
            # print()
        else:
            categs_counts = None
        return N, num_violent, categs_counts

VW_WORD_COL = 'word'
VW_GROUP_COL = 'group'
VW_CATEG_COL = 'categ'

def load_violent_words(lang, as_dframe=False) -> List[str]:
    if lang == 'es': fpath = VIOLENT_WORDS_ES
    elif lang == 'en': fpath = VIOLENT_WORDS_EN
    else: raise ValueError(f'lang {lang} not supported')
    df = pd.read_excel(fpath, header=0, names=[VW_WORD_COL, VW_GROUP_COL, VW_CATEG_COL])
    if as_dframe: return df
    words = list(df[df[VW_GROUP_COL].isin([1, 2])][VW_WORD_COL])
    return words

def load_violent_categ_map(lang) -> Dict[str, str]:
    df = load_violent_words(lang, as_dframe=True)
    categ_map = { row[VW_WORD_COL]: row[VW_CATEG_COL] for _, row in df.iterrows() }
    return categ_map

VIOLENT_CATEGS = ['war', 'rebellion', 'crime', 'destruction', 'death', 'disqualification or insult',
                  'danger', 'injustice/immoral', 'manipulation', 'chaos', 'punishment', 'enemy']

def find_string_with_edit_distance(S, lst):
    close_matches = [s for s in lst if editdistance.eval(S, s) <= 2]
    if len(close_matches) == 0:
        return None
    elif len(close_matches) == 1:
        return close_matches[0]
    else:
        raise ValueError("More than one string found with edit distance <= 2")

def check_violence_categories(lang):
    df = load_violent_words(lang, as_dframe=True)
    # print unique categories
    print(df[VW_CATEG_COL].unique())

if __name__ == '__main__':
    #print(load_violent_words('en'))
    check_violence_categories('es')