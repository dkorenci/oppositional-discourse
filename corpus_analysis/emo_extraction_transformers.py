import pandas
from data_tools.classif_data.data_loaders import corpus_telegram_en_ph1
from transformers import pipeline

from corpus_analysis.span_categs_data_create import create_spanfeatures_dataset


def hf_txt_classif2map(pred, labels=None):
    ''' Convert a HuggingFace text classification prediction to a map label -> score.
    HF prediction is a list of dicts with keys 'label' and 'score'.
    :param pred: HF prediction
    :param labels: if not None, only labels in this set are returned
    :return: map label -> score
    '''
    if labels is not None:
        labels = set(labels)
        return {d['label']: d['score'] for d in pred if d['label'] in labels}
    else: # all labels
        return {d['label'] : d['score'] for d in pred}

def apply_hf_classifier_to_df(df, model_name, labels=None, text_column='text', id_column = 'text_id', test=None,
                              batch_size=8):
    '''
    Applies a HuggingFace classifier to texts in a dataframe, and returns a map id -> classification,
    where classification is a map category -> score.
    '''
    if test is not None: df = df[:test]
    classifier = pipeline("text-classification", model=model_name, return_all_scores=True,
                          truncation=True, batch_size=batch_size)
    res = classifier(df[text_column].values.tolist())
    ids = df[id_column].values.tolist()
    id2class = {}
    if labels is not None:
        labels = set(labels)
        given_labels = True
    else: given_labels = False
    for i in range(len(ids)):
        id2class[ids[i]] = hf_txt_classif2map(res[i], labels=labels)
        if labels is None: labels = set(id2class[ids[i]].keys())
        else:
            if given_labels: assert labels.issubset(set(id2class[ids[i]].keys()))
            else: assert set(id2class[ids[i]].keys()) == labels
    return id2class

def emo_test():
    df = corpus_telegram_en_ph1()
    res = apply_hf_classifier_to_df(df, model_name='j-hartmann/emotion-english-roberta-large', test=200)
    #res = apply_hf_classifier_to_df(df, model_name='j-hartmann/emotion-english-distilroberta-base', test=200)
    #res = apply_hf_classifier_to_df(df, model_name='siebert/sentiment-roberta-large-english', test=200)
    print(res)


def add_prefixed_features_to_df(df: pandas.DataFrame, pfeatures: dict, id_column='text_id'):
    '''
    :param df: a dataframe with an id column
    :param pfeatures: map perfix -> map (id -> features (label-> score))
    :return: a dataset with new column prefix_label, for each prefix in param and label in features
    '''
    for prefix, id2features in pfeatures.items():
        labels = list(id2features[list(id2features.keys())[0]].keys())
        for label in labels:
            df[prefix + '_' + label] = df[id_column].apply(lambda x: id2features[x][label])
    return df

def add_features_to_df(df: pandas.DataFrame, id2features: dict, id_column='text_id', all_features=None, default_value=None):
    '''    
    :param df: a dataframe with an id column 
    :param id2features: map (id -> features (label-> score))     
    :param all_features: list of all features (new df columns)
    :param default_value: value for features not given for a current row
    :return: 
    '''
    if all_features is not None:
        for feature in sorted(all_features):
            df[feature] = default_value
    for id, feature_values in id2features.items():
        for feature, value in feature_values.items():
            df.loc[df[id_column] == id, feature] = value
    return df

def add_emo_transformer_features_to_df(df, lang='en', test=None):
    if test is not None: df = df[:test]
    if lang == 'en':
        emores = apply_hf_classifier_to_df(df, model_name='j-hartmann/emotion-english-distilroberta-base', batch_size=128,
                                       labels=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'])
    elif lang == 'es':
        emores = apply_hf_classifier_to_df(df, model_name='maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1',
                                           batch_size=128, labels=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'])
    df = add_features_to_df(df, emores)
    return df

if __name__ == '__main__':
    #emo_test()
    #create_english_emo_dataset()
    add_emo_transformer_features_to_df(create_spanfeatures_dataset(lang='en', save=False), lang='es', test=10)