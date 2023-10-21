import pandas as pd
import pickle

from classif_experim.classif_experiment_runner import run_classif_experiments
from data_tools.classif_data.data_utils import create_classif_dataset
from data_tools.seqlab_data.labstudio_convert import TEXT_ID_FIELD_NAME

BEST_BINARY_MODELS = {
    'en': ['microsoft/deberta-v3-base'],
    'es': ['dccuchile/bert-base-spanish-wwm-cased'],
}

def generate_prediction_dataset(lang, labels=['A', 'F', 'P', 'V'], test=False):
    col_names = [TEXT_ID_FIELD_NAME] + labels
    df = pd.DataFrame(columns=col_names)
    preds = { l: None for l in labels }
    for label in labels:
        res = run_classif_experiments(lang, num_folds=5, experim_type=f'spancateg-{label}', test=test,
                                experim_label='cvpredict', model_list=BEST_BINARY_MODELS[lang], rnd_seed=3561467,
                                pause_after_fold=0, pause_after_model=0)
        best_model = BEST_BINARY_MODELS[lang][0]
        preds[label] = res[best_model] # map ID -> binary class prediction
    # serialize to disk the preds object, tag the file with lang
    with open(f'{lang}.span.categ.predict.pkl', 'wb') as f:
        pickle.dump(preds, f)
    # test that for each label, the predictions are on the same set of ids
    all_ids = set(preds[labels[0]].keys())
    _, _, orig_classif_ids = create_classif_dataset(lang, output='sklearn')
    if not all_ids == set(orig_classif_ids):
        raise ValueError(f'Prediction ids and original classif. ids are not the same!')
    for label in labels:
        if not preds[label].keys() == all_ids:
            raise ValueError(f'Prediction sets for labels {label} and {labels[0]} are not the same!')
    # create dataframe
    for text_id in all_ids:
        row = [text_id] + [preds[label][text_id] for label in labels]
        df = df.append(pd.Series(row, index=col_names), ignore_index=True)
    df.to_excel(f'{lang}.span.categ.predict.xlsx', index=False)

if __name__ == '__main__':
    generate_prediction_dataset('en', test=False)
    generate_prediction_dataset('es', test=False)