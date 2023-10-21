'''
Export of texts with span-level features, with added emotion features.
'''
from corpus_analysis.emo_extraction_liwc import add_liwc_features_to_df
from corpus_analysis.emo_extraction_transformers import apply_hf_classifier_to_df, add_prefixed_features_to_df, \
    add_emo_transformer_features_to_df
from corpus_analysis.violence_extract.violent_words_feat_extractor import add_violent_words_features_to_df
from corpus_analysis.span_categs_data_create import create_spanfeatures_dataset

def create_english_emo_dataset_v1(test=None):
    df = create_spanfeatures_dataset(lang='en')
    if test is not None: df = df[:test]
    sentires = apply_hf_classifier_to_df(df, model_name='siebert/sentiment-roberta-large-english',
                                         labels=['NEGATIVE', 'POSITIVE'], batch_size=128)
    emores = apply_hf_classifier_to_df(df, model_name='j-hartmann/emotion-english-roberta-large', batch_size=128,
                                       labels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    df = add_prefixed_features_to_df(df, {'senti': sentires, 'emo': emores})
    df.to_excel(f'en.emosenti.dataset.xlsx', index=False)

def create_emo_dataset_full(lang='en', test=None):
    df = create_spanfeatures_dataset(lang=lang)
    if test: df = df[:test]
    # get emo features
    print('creating emo-transformer features...')
    df = add_emo_transformer_features_to_df(df, lang=lang)
    # get liwc emo features
    print('creating liwc features...')
    df = add_liwc_features_to_df(df, lang=lang)
    # save the dataset
    print('saving the dataset...')
    df.to_excel(f'{lang}.emo.liwc.dataset.xlsx', index=False)

def create_violence_dataset(lang='en', test=None, gold=False, verbose=True):
    df = create_spanfeatures_dataset(lang=lang, gold=gold, verbose=verbose)
    if test: df = df[:test]
    df = add_violent_words_features_to_df(df, lang=lang, categs=True)
    df = add_liwc_features_to_df(df, lang=lang)
    print('saving the dataset...')
    df.to_excel(f'{lang}.violence.dataset.xlsx', index=False)

def export_spanfeatures_dataset(lang, verbose=True):
    df = create_spanfeatures_dataset(lang, verbose=verbose)
    df.to_excel(f'{lang}.dataset.xlsx', index=False)

if __name__ == '__main__':
    #create_emo_dataset_full(lang='es', test=None)
    #print(create_spanfeatures_dataset('es'))
    create_violence_dataset(lang='en', test=None, gold=True, verbose=True)