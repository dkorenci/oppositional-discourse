from corpus_analysis.violence_extract.violence_index_calc import load_violent_words, VW_CATEG_COL


def print_word_categories(lang):
    wdf = load_violent_words(lang, as_dframe=True)
    wdf[VW_CATEG_COL][wdf[VW_CATEG_COL].isna()] = 'none'
    print(wdf[VW_CATEG_COL].value_counts())
    print()
    print(wdf[VW_CATEG_COL].describe())
    # iter over name, count pairs in wdf[VW_CATEG_COL].value_counts()
    N = len(wdf)
    categ_counts = [ (name, count) for name, count in wdf[VW_CATEG_COL].value_counts().iteritems() ]
    categ_counts.sort(key=lambda x: x[1], reverse=True)
    for name, count in wdf[VW_CATEG_COL].value_counts().iteritems():
        print(f'{name}: {count/N*100:.3f}%')


if __name__ == '__main__':
    print_word_categories('en')
