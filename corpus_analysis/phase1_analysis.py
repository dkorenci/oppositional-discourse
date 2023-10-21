'''
Analysis of "phase 1" corpora annotated binary for conspiratorial language and critical thinking language.
'''
import pandas as pd

from corpus_analysis.plotting_utils import *
from data_tools.classif_data.data_utils import *

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

def word_cnt_dist(df):
    print(df[NUM_WORDS].describe())
    fig, ax = plt.subplots(1, 2)
    sns.histplot(df, x=NUM_WORDS, ax=ax[0])
    sns.boxplot(df, y=NUM_WORDS, ax=ax[1])
    plt.tight_layout()
    plt.show()
    #plt.savefig('word_count_dist.pdf')

def word_cnt_compare(df, plot='hist'):
    add_class_labels(df)
    fig, ax = plt.subplots()
    if plot == 'box':
        sns.boxplot(df, x=CLS_TRNRY_LAB, y=NUM_WORDS, ax=ax)
    elif plot == 'hist':
        col = ['green', 'blue', 'red']
        for c in TRNRY_CLASSES:
            sns.histplot(df[df[CLS_TRNRY_IX] == c], x=NUM_WORDS, ax=ax, color=col[c])
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()

def print_shortest_texts(df, N = 50):
    df['txt_len'] = df[TXT_COLUMN].apply(lambda x: len(str(x)))
    for ix, r in df.sort_values('txt_len').iterrows():
        print(f'id: {r["id"]}, txt: [{r[TXT_COLUMN]}]')
        N -= 1
        if N == 0: break

def text_label_stats(df):
    N = len(df)
    # basic counts
    print(f'Conspiracy: {sum(df[GOLD_CONSPI]):4}, non-conspiracy: {N-sum(df[GOLD_CONSPI]):4}')
    print(f'  Critical: {sum(df[GOLD_CRITIC]):4},   non-critical: {N-sum(df[GOLD_CRITIC]):4}')
    print(f'      none: {N - sum(df[GOLD_CONSPI] | df[GOLD_CRITIC])}')
    print(f'      both: {sum(df[GOLD_CONSPI] & df[GOLD_CRITIC])}')
    #sns.barplot(df, y=GOLD_CONSPI)
    CON_SUM, CRT_SUM = 'ANNOT_CONSPI_SUM', 'ANNOT_CRITIC_SUM'
    df[CON_SUM] = df[ANNOT_CONSPI[1:]].sum(axis=1)
    df[CRT_SUM] = df[ANNOT_CRITIC[1:]].sum(axis=1)
    print('Conspiracy annotator scores, over entire dataset:')
    print(df[CON_SUM].value_counts(normalize=True).sort_index())
    print('Conspiracy annotator scores, over conspiratorial (gold = 1):')
    print(df[CON_SUM][df[GOLD_CONSPI]==1].value_counts(normalize=True).sort_index())
    print()
    print('Critical annotator scores, over entire dataset:')
    print(df[CRT_SUM].value_counts(normalize=True).sort_index())
    print('Critical annotator scores, over critical (gold = 1):')
    print(df[CRT_SUM][df[GOLD_CRITIC]==1].value_counts(normalize=True).sort_index())
    #sns.histplot(df, x=CON_SUM)
    #plt.show()

def plot_text2d(df, rndSeed=82271):
    '''
    Dim-reduce text to 2d, and plot labels as different colors.
    :return:
    '''
    add_class_labels(df)
    #print(df[CLS_TRNRY].value_counts())
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,1))
    # convert to string because 1 text is '1' - it's an error
    texts = list(str(txt) for txt in df[TXT_COLUMN])
    txt_vecs = tfidf.fit_transform(texts)
    dimReduce = TSNE(random_state=rndSeed)
    txt2d = dimReduce.fit_transform(txt_vecs)
    plotdf = pd.DataFrame({'x': txt2d[:, 0], 'y': txt2d[:, 1],
                           'class': df[CLS_TRNRY_LAB], 'num_words': df[NUM_WORDS]})
    sns.scatterplot(plotdf, x='x', y='y', hue='class', size='num_words', alpha=0.65)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    plt.savefig('texts2d.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    #CORPUS = corpus_telegram_es_ph1()
    CORPUS = corpus_telegram_en_ph1()
    word_cnt_dist(CORPUS)
    text_label_stats(CORPUS)
    #plot_text2d(CORPUS)
    #word_cnt_compare(ES_CORPUS)
    #print_shortest_texts(ES_CORPUS)