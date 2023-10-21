from functools import partial

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef

f1_macro = partial(f1_score, average='macro')
f1_binary = partial(f1_score, average='binary')

def classif_scores(setup='binary'):
    f1_macro = partial(f1_score, average='macro')
    f1_binary = partial(f1_score, average='binary')
    if setup == 'binary':
        score_fns = {'F1': f1_binary, 'ACC': accuracy_score,
                     'prec': precision_score, 'recall': recall_score}
    elif setup == 'multiclass':
        score_fns = { 'F1_macro': f1_macro, 'ACC':accuracy_score, 'MCC': matthews_corrcoef }
    elif setup == 'all':
        score_fns = {'F1_macro': f1_macro, 'F1': f1_binary, 'ACC': accuracy_score,
                     'prec': precision_score, 'recall': recall_score, 'MCC': matthews_corrcoef}
    elif setup == 'span-binary':
        score_fns = {'F1m': f1_macro, 'F1': f1_binary, 'P': precision_score, 'R': recall_score}
    return score_fns
