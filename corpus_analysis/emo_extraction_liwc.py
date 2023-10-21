import liwc
import spacy
from liwc.dic import _parse_categories

from corpus_analysis.emo_extraction_transformers import add_prefixed_features_to_df, add_features_to_df
from corpus_analysis.span_categs_data_create import create_spanfeatures_dataset
from settings import LWIC_ENGLISH_DICT_PATH, LWIC_SPANISH_DICT_PATH

def liwc_categ_label2code(filepath):
    '''
    Mapping of category labels to category codes
    :param filepath: a LWIC .dic file
    :return: map token -> category
    '''
    # code taken from liwc.dic.read_dic, works for lwic 0.5.0
    with open(filepath) as lines:
        for line in lines:
            if line.strip() == "%":
                break
        category_mapping = dict(_parse_categories(lines)) # categories -> words
    return {v: k for k, v in category_mapping.items()}

def load_liwc_processing(filepath):
    '''
    Return word parser (callable token -> [categ. labels]) and label2code mapping.
    :param filepath: a LWIC .dic file
    :return:
    '''
    wparse, category_names = liwc.load_token_parser(filepath)
    label2code = liwc_categ_label2code(filepath)
    assert set(label2code.keys()) == set(category_names)
    return wparse, label2code

class LIWCTextParser():

    def __init__(self, wparse, label2code, lang, output='code'):
        self.wparse, self.label2code, self.output = wparse, label2code, output
        self.lang = lang
        self.nlp = spacy.blank(lang)

    def __call__(self, text):
        '''
        :param text: string
        :return: num tokens, map category code -> num. of occurences
        '''
        doc = self.nlp(text)
        num_tok = len(doc)
        res = {}
        for token in doc:
            for label in self.wparse(token.text.lower()):
                key = self.label2code[label] if self.output == 'code' else label
                res[key] = res.get(key, 0) + 1
        return num_tok, res

def get_liwc_parser(lang, output='code'):
    if lang == 'en': wparse, label2code = load_liwc_processing(LWIC_ENGLISH_DICT_PATH)
    elif lang == 'es': wparse, label2code = load_liwc_processing(LWIC_SPANISH_DICT_PATH)
    else: raise ValueError(f'Unknown language {lang}')
    return LIWCTextParser(wparse, label2code, lang, output)

def get_all_liwc_categs(lang, output='code'):
    if lang == 'en': wparse, label2code = load_liwc_processing(LWIC_ENGLISH_DICT_PATH)
    elif lang == 'es': wparse, label2code = load_liwc_processing(LWIC_SPANISH_DICT_PATH)
    else: raise ValueError(f'Unknown language {lang}')
    if output == 'code': return list(set(label2code.values()))
    else: return list(set(label2code.keys()))

def liwc_test():
    wparse, label2code = load_liwc_processing(LWIC_ENGLISH_DICT_PATH)
    #wparse, label2code = load_lwic_processing(LWIC_ENGLISH_DICT_PATH)
    for w in ['administrator', 'administration', 'administrat', 'administra']:
    #for w in ['cabron', 'puta', 'amor', 'hijo']:
        print(f'WORD: {w}')
        for category in wparse(w):
            print('\t', category, label2code[category])

def liwc_parser_test(lang, output='code'):
    parser = get_liwc_parser(lang, output)
    for text in ['I am happy', 'I am sad', 'I am stupid']:
        print(f'TEXT: {text}')
        N, categs = parser(text)
        print(f'\tN: {N}')
        print(';'.join(f'{categ}: {count}' for categ, count in categs.items()))

def normalize_liwc_features(df, lang):
    '''
    Normalize LWIC features by the number of words in the text
    :param df: dataframe with features
    :return: dataframe with normalized features
    '''
    liwc_features = get_all_liwc_categs(lang, output='code')
    for feature in liwc_features:
        df[feature] = df[feature] / df['num_words'] * 100
    return df

def apply_liwc_extractor_to_df(df, lwic_parser, text_column='text', id_column ='text_id'):
    res = {}
    for i, row in df.iterrows():
        text_id = row[id_column]
        N, categs = lwic_parser(row[text_column])
        res[text_id] = {'num_words': N, **categs}
    return res

def add_liwc_features_to_df(df, lang='en', test=None, text_column='text', id_column ='text_id'):
    parser = get_liwc_parser(lang, output='code')
    if test is not None: df = df.head(test)
    liwc_res = apply_liwc_extractor_to_df(df, parser, text_column, id_column)
    # add param - a list with all the labels, and default value for these columns
    all_categs = get_all_liwc_categs(lang, output='code')
    all_categs.sort(key=lambda x: int(x))
    add_features_to_df(df, liwc_res, id_column=id_column, all_features=all_categs, default_value=0)
    return df

if __name__ == '__main__':
    liwc_test()
    #lwic_parser_test('en', output='label')
    #add_liwc_features_to_df(create_spanfeatures_dataset(lang='en', save=False), lang='en', test=10)