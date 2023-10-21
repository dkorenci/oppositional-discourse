from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from corpus_analysis.emo_extraction_liwc import add_liwc_features_to_df, normalize_liwc_features
from corpus_analysis.span_categs_data_create import get_df, CONSPI_CRITIC, create_spanfeatures_dataset, SPAN_PROPERTY, VALUE, \
    SPAN_CATEGORY, AnnotFeatExtractor


def cmp_text_size_plot():
    ''' Side by side text length boxplots for conspiracy/critical. '''
    for lang in ['en', 'es']:
        df = get_df(lang, text_level_only=True)
        matplotlib.use('TkAgg')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=CONSPI_CRITIC, y='num_words', data=df, ax=ax, order=['critical', 'conspiracy'])
        sns.despine(offset=10, trim=True)
        # plot formatting
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='dashed')
        title = f'Text size distributions for text types , language = {lang}'
        plt.title(title)
        plt.ylabel('number of words in text')
        plt.tight_layout()
        plt.savefig(f'{lang}.text.size.pdf', dpi=500)


def sidebyside_property_boxplots(df, lang, property, ylabel, ylimit=None):
    df = df[df[SPAN_PROPERTY] == property]
    df[property] = df[VALUE]
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(12, 6))
    if ylimit:
        plt.ylim(0, ylimit)
    sns.boxplot(x=CONSPI_CRITIC, y=property, hue=SPAN_CATEGORY, data=df, ax=ax,
                order=['critical', 'conspiracy'])
    sns.despine(offset=10, trim=True)
    # plot formatting
    if ylimit:
        if ylimit <= 40: ax.set_yticks(list(range(0, ylimit)))
        else: ax.set_yticks(list(range(0, ylimit, 10)))
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='dashed')
    title = f'{ylabel}, for all span categories and text types, language = {lang}'
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'{lang}.{property}.pdf', dpi=500)

def count_for_categs(ylimit={'en': 30, 'es': 30}):
    for lang in ['en', 'es']:
        df = get_df(lang)
        sidebyside_property_boxplots(df, lang, AnnotFeatExtractor.CNT_LABEL,
                                     ylabel='number of spans per text', ylimit=ylimit[lang])

def txtperc_for_categs(ylimit={'en': 80, 'es': 80}):
    for lang in ['en', 'es']:
        df = get_df(lang)
        sidebyside_property_boxplots(df, lang, AnnotFeatExtractor.SIZE_PERC_LABEL,
                                     ylabel='percentage of text within all spans', ylimit=ylimit[lang])

def emo_vs_critconspi_plot(lang, dset_file, emo_property, ylabel=''):
    # load data from file
    # plot: emo_feature distribution vs. conspiracy/critical
    # plot: emo_feature distribution for each BINARY span category, vs. conspiracy/critical
    #df = create_spanfeatures_dataset(lang)
    #df = add_liwc_features_to_df(df, lang=lang)
    df = pd.read_excel(dset_file)
    normalize_liwc_features(df, lang)
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=CONSPI_CRITIC, y=emo_property, data=df, ax=ax, order=['critical', 'conspiracy'])
    sns.despine(offset=10, trim=True)
    # plot formatting
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='dashed')
    title = f'{emo_property} size distributions for text types , language = {lang}'
    plt.title(title)
    if not ylabel: ylabel = emo_property
    plt.ylabel(f'{ylabel}')
    plt.tight_layout()
    plt.savefig(f'{lang}.{emo_property}.critic.vs.conspi.pdf', dpi=500)
    print(df.columns)

def replace_num_words(dset_file, col_name):
    ''' remove the 'num_words' column and rename col_name to 'num_words' '''
    df = pd.read_excel(dset_file)
    df.drop(columns=['num_words'], inplace=True)
    df.rename(columns={col_name: 'num_words'}, inplace=True)
    df.to_excel(dset_file, index=False)

def annot_level_aggregate(file):
    df = pd.read_excel(file)
    print(df.columns)
    df['AGG'] = -1
    for text_id in set(df['text_id']):
        text_df = df[df['text_id'] == text_id]
        # assert there are exectly two rows for the id, with distinct 'annotator' values
        assert len(text_df) == 2
        assert len(set(text_df['annotator'])) == 2
        type_conflict = set(text_df['Type_conflict'])
        # if there is only one value, assign to 1 to 'AGG' column for both rows, else assign 0
        if len(type_conflict) == 1:
            df.loc[text_df.index, 'AGG'] = 1
        else:
            df.loc[text_df.index, 'AGG'] = 0
    assert set(df['AGG']) == {0, 1}
    df.to_excel(file.replace('.xlsx', '.agg.xlsx'), index=False)

if __name__ == '__main__':
    emo_vs_critconspi_plot('es', 'es.violence.dataset.xlsx', emo_property='violence_index', ylabel='violence')
