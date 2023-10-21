from corpus_analysis.span_categs_data_create import NUM_WORDS_COLUMN
from corpus_analysis.violence_extract.violence_index_calc import ViolenceIndexCalculator, VIOLENT_CATEGS


def add_violent_words_features_to_df(df, lang='en', update_length=False, categs=False):
    '''
    Calculate and add to df the number of violent words in each text,
    and the number of word per category of violent words.
    :param update_length: update the number of words in text with data from ViolenceIndexCalculator's nlp object
            instead of using whitespace tokenization
    '''
    violence_calc = ViolenceIndexCalculator(lang)
    res = [violence_calc.violent_idx_data(txt, categs=categs) for txt in df['text'].values.tolist()]
    df['num_violent_words'] = [r[1] for r in res]
    if update_length: df[NUM_WORDS_COLUMN] = [r[0] for r in res]
    if categs:
        for categ in VIOLENT_CATEGS: df[categ] = 0
        for idx, r in enumerate(res):
            txt_categs = r[2]
            for categ, count in txt_categs.items(): df.at[idx, categ] = count
    return df
