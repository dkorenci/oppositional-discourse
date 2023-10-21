from classif_experim.classif_experiment_runner import HF_MODEL_LIST, build_transformer_model

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.classif_data.data_loaders import classif_corpus_raw_df
from data_tools.classif_data.data_utils import TXT_COLUMN
from data_tools.raw_dataset_loaders import load_raw_datasets


def tokenize_and_count(texts, tokenizer):
    """
    Tokenizes a list of texts and returns the number of tokens for each text.
    """
    token_counts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_counts.append(len(tokens))
    return token_counts


def five_number_summary(data):
    """
    Returns the five-number summary (min, Q1, median, Q3, max) for a list of data.
    """
    return {
        "min": np.min(data),
        "Q1": np.percentile(data, 25),
        "median": np.median(data),
        "Q3": np.percentile(data, 75),
        "max": np.max(data)
    }


def do_analysis(W, data):
    """
    Tokenizes, calculates statistics, and plots the data.

    W: a dictionary with labels as keys and wrappers with .model and .tokenizer properties as values.
    data: a dictionary with labels as keys and lists of texts as values.
    """
    all_token_counts = {}  # { label -> list of token counts }
    for label, texts in data.items():
        tokenizer = W[label].tokenizer
        token_counts = tokenize_and_count(texts, tokenizer)
        all_token_counts[label] = token_counts

        print(f"Statistics for {label}:")
        stats = five_number_summary(token_counts)
        for k, v in stats.items():
            print(f"{k}: {v}")
        print()

    # Plotting
    all_labels = list(all_token_counts.keys())
    all_counts = [all_token_counts[label] for label in all_labels]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=all_counts)
    plt.xticks(range(len(all_labels)), all_labels)
    plt.ylabel('Number of Tokens')
    plt.title('Box and Whiskers Plot for Token Counts')
    plt.show()



def run_toknization_analysis(lang):
    corpus = classif_corpus_raw_df(lang)
    corpus = list(corpus[TXT_COLUMN])
    models, dsets = {}, {}
    for model_label in HF_MODEL_LIST[lang]:
        model = SklearnTransformerClassif(model_label, lang=lang)
        model._init_model(2)
        models[model_label] = model
        dsets[model_label] = corpus
    do_analysis(models, dsets)

if __name__ == '__main__':
    run_toknization_analysis('en')
