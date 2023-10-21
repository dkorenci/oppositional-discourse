import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from classif_experim.classif_utils import classif_scores
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.classif_data.data_utils import *

ES_CORPUS = corpus_telegram_es_ph1()
add_class_labels(ES_CORPUS)

def corpus(lang):
    if lang.lower() == 'es': c = corpus_telegram_es_ph1()
    elif lang.lower() == 'en': c = corpus_telegram_en_ph1()
    add_class_labels(c)
    return c


def fextr_tfidf(ngram_max=1):
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram_max))

def fextr_nmf(num_topics, rnd_seed):
    return Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,1))),
                     ('nmf', NMF(n_components=num_topics, init='nndsvd', random_state=rnd_seed))])

def fextr_tfidfnmf(num_topics, rnd_seed, ngram_max=1):
    return FeatureUnion([('tfidf', fextr_tfidf(ngram_max)), ('nmf', fextr_nmf(num_topics, rnd_seed))])

def build_model(model_label, features, lang, cls_weight, rnd_seed, test=False, model_options=None, ):
    np.random.seed(rnd_seed) # global for sklearn
    # features
    if features == 'tfidf': fextr = fextr_tfidf()
    elif features == 'nmf': fextr = fextr_nmf(100, rnd_seed)
    elif features == 'tfidf-nmf': fextr = fextr_tfidfnmf(100, rnd_seed)
    elif features == None: fextr = None # model that works with raw text
    else: raise ValueError(f'Unknown features: {features}')
    # model
    if model_label == 'logreg': classif = LogisticRegression(random_state=rnd_seed, class_weight=cls_weight)
    elif model_label == 'lin_svm': classif = LinearSVC(random_state=rnd_seed, class_weight=cls_weight)
    elif model_label == 'xgb': classif = XGBClassifier(random_state=rnd_seed)
    elif model_label == 'transformer':
        # default values for model options' params, if they are not specified
        epochs, eval = (3, 0.2) if not test else (2, 0.5)
        model_options['num_train_epochs'] = model_options.get('epochs', epochs)
        model_options['eval'] = model_options.get('eval', eval)
        model_options['lang'] = model_options.get('lang', lang)
        classif = SklearnTransformerClassif(**model_options, rnd_seed=rnd_seed)
    else: raise ValueError(f'unknown model label: {model_label}')
    if features is None: return classif
    else: return Pipeline([('fextr', fextr), ('classifier', classif)])

def run_crossvalid(lang, model_label, use_classes='all', features='tfidf', cls_weight=None,
                   num_folds=5, rnd_seed=3154561, test=False, model_options=None, binary=False):
    '''
    Measure classification performance of a txt classification model on x-valid folds.
    :param dset:
    :param model:
    :param num_folds:
    :param rnd_seed:
    :param target:
    :return:
    '''
    score_fns = classif_scores('binary')
    texts, classes, _ = create_classif_dataset(lang, classes=use_classes, output='sklearn')
    if test: texts, classes = texts[:test], classes[:test]
    #texts, classes = np.array(texts), np.array(classes)
    # todo create 2*num_folds splits, by using strat-k-fold twice, with a separate simple wrapper class
    # alternatively, use shuffle split?
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    results_df = pd.DataFrame(columns=score_fns.keys())
    conf_mx = None; rseed = rnd_seed
    for train_index, test_index in foldgen.split(texts, classes):
        model = build_model(model_label, features, lang=lang, model_options=model_options,
                            cls_weight=cls_weight, rnd_seed=rseed, test=test); rseed += 1
        print(model)
        fold_index += 1
        # split data
        txt_tr, txt_tst = texts[train_index], texts[test_index]
        cls_tr, cls_tst = classes[train_index], classes[test_index]
        # train model
        model.fit(txt_tr, cls_tr)
        # evaluate model
        cls_pred = model.predict(txt_tst)
        del model
        scores = pd.DataFrame({fname: [f(cls_tst, cls_pred)] for fname, f in score_fns.items()})
        results_df = pd.concat([results_df, scores], ignore_index=True)
        conf_mx_tmp = confusion_matrix(cls_tst, cls_pred)
        if conf_mx is None: conf_mx = conf_mx_tmp
        else: conf_mx += conf_mx_tmp
    conf_mx = conf_mx.astype('float64')
    conf_mx /= num_folds
    print(f'CLASSIFIER: {model_label:10}; {str(features):10}, CLASSES = {";".join(use_classes)}')
    for fname in score_fns.keys():
        print(f'{fname:10}: ', '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    print('Confusion matrix:')
    for r in conf_mx:
        print(', '.join(f'{v:7.2f}' for v in r))
    #print(results_df)


def run_crossvalid_multiling(model_label, langs=['en', 'es'], use_classes='all', features='tfidf', cls_weight=None,
                   num_folds=5, rnd_seed=3154561, test=False, model_options=None, binary=False):
    '''
    Measure classification performance of a txt multi-lingual classification model on x-valid folds.
    :param dset:
    :param model:
    :param num_folds:
    :param rnd_seed:
    :param target:
    :return:
    '''
    score_fns = classif_scores('binary')
    texts, classes = {}, {}
    for lang in langs:
        texts[lang], classes[lang], _ = create_classif_dataset(lang, classes=use_classes, output='sklearn')
        if test: texts[lang], classes[lang] = texts[lang][:test], classes[lang][:test]
    # check data consistency across languages
    NC = None
    for lang in langs:
        assert len(texts[lang]) == len(classes[lang])
        if NC is None: NC = len(set(classes[lang]))
        assert len(set(classes[lang])) == NC
    N = len(texts[langs[0]]) # number of samples
    # prepare classification folds and results data structures
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    results_df, conf_mx = {}, {}
    for lang in langs:
        results_df[lang] = pd.DataFrame(columns=score_fns.keys())
        conf_mx[lang] = None
    rseed = rnd_seed
    # evaluate
    for train_index, test_index in foldgen.split(texts[langs[0]], classes[langs[0]]):
        model = build_model(model_label, features, model_options=model_options,
                            cls_weight=cls_weight, rnd_seed=rseed, test=test);
        rseed += 1
        fold_index += 1
        # take train splits for each language, and merge them into a single train set
        txt_tr, cls_tr = [], []
        for lang in langs:
            txt_tr.extend(texts[lang][train_index])
            cls_tr.extend(classes[lang][train_index])
        # train model
        model.fit(txt_tr, cls_tr)
        # evaluate model, for each language
        for lang in langs:
            txt_tst, cls_tst = texts[lang][test_index], classes[lang][test_index]
            cls_pred = model.predict(txt_tst)
            scores = pd.DataFrame({fname: [f(cls_tst, cls_pred)] for fname, f in score_fns.items()})
            results_df[lang] = pd.concat([results_df[lang], scores], ignore_index=True)
            conf_mx_tmp = confusion_matrix(cls_tst, cls_pred)
            if conf_mx[lang] is None: conf_mx[lang] = conf_mx_tmp
            else: conf_mx[lang] += conf_mx_tmp
        del model
    # print results
    for lang in langs:
        conf_mx[lang] = conf_mx[lang].astype('float64')
        conf_mx[lang] /= num_folds
    print(f'CLASSIFIER: {model_label:10}; {str(features):10}, CLASSES = {";".join(use_classes)}')
    for lang in langs:
        print(f'RESULTS FOR LANGUAGE: {lang}')
        for fname in score_fns.keys():
            print(f'{fname:10}: ', '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[lang][fname].describe().items()))
        print('Confusion matrix:')
        for r in conf_mx[lang]:
            print(', '.join(f'{v:7.2f}' for v in r))
        print()

def run_crossvalid_for_models(lang='es', features='tfidf'):
    for classif in ['logreg', 'xgb', 'lin_svm']:
        run_crossvalid(corpus(lang), model_label=classif, features=features, use_classes=[LBL_CRITIC, LBL_CONSP])

if __name__ == '__main__':
    #for model in ['xgb', 'logreg', 'lin_svm']: run_crossvalid(ES_CORPUS, model, features='tfidf-nmf')
    #run_crossvalid_for_models('en', 'nmf')
    run_crossvalid('en', 'transformer', model_options= {'eval': 0.2, 'hf_model_label': 'roberta-base', #'microsoft/deberta-v3-base',
                                                        'batch_size': 32, 'gradient_accumulation_steps': 1, },
                   test=None, features=None, use_classes=[LBL_CONSP, LBL_CRITIC])
    # run_crossvalid('es', 'transformer',
    #                model_options= {'eval': None, 'model_label': 'dccuchile/bert-base-spanish-wwm-cased', # 'PlanTL-GOB-ES/roberta-base-bne', #
    #                                'batch_size': 16, 'gradient_accumulation': 2},
    #                test=None, features=None, use_classes=[LBL_CRITIC, LBL_CONSP])
    #run_crossvalid('en', 'logreg', features='tfidf', use_classes='all')
    # run_crossvalid(ES_CORPUS, 'logreg', features='tfidf', use_classes=[LBL_CRITIC, LBL_NOCONSP],
    #                cls_weight='balanced')
    # run_crossvalid_multiling(model_label='transformer', model_options=
    #                             {'eval': None, 'model_label': 'bert-base-multilingual-cased', 'epochs': 5,
    #                              'batch_size': 16, 'gradient_accumulation': 2},
    #                          test=None, features=None, use_classes=[LBL_CRITIC, LBL_CONSP])
    # run_crossvalid_multiling(model_label='logreg',
    #                model_options= {'eval': None, 'model_label': 'bert-base-multilingual-cased', 'epochs': 1},
    #                test=None, features='tfidf', use_classes=[LBL_CRITIC, LBL_CONSP])