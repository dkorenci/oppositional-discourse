'''
Classes for working with hugginface models.
'''
import datasets
import torch
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, \
                            TextClassificationPipeline

from classif_experim.classif_utils import classif_scores
from classif_experim.pynvml_helpers import print_gpu_utilization
from data_tools.classif_data.data_loaders import *
from data_tools.classif_data.data_utils import *


def build_model(lang, model_label, classes='all', max_length=None, epochs=5, rnd_seed=391772):
    '''
    Text classification with transformers from hugginface.
    Derived from: https://huggingface.co/docs/transformers/tasks/sequence_classification
    '''
    print_gpu_utilization('start')
    if lang == 'en': df = corpus_telegram_en_ph1()
    elif lang == 'es': df = corpus_telegram_es_ph1()
    else: raise ValueError('supported languages are "en" and "es"')
    df = add_class_labels(df)
    if classes != 'all':
        df = binary_dataset(df, classes)
        num_classes = 2
    else: num_classes = 3
    df['label'] = df[CLS_TRNRY_IX]
    train, dev = train_test_split(df, test_size=0.2, random_state=rnd_seed, stratify=df[[CLS_TRNRY_IX]])
    dset = DatasetDict({'train': datasets.Dataset.from_pandas(train), 'test': datasets.Dataset.from_pandas(dev)})
    tokenizer = AutoTokenizer.from_pretrained(model_label)
    tokenizer_params = {'truncation': True}
    if max_length is not None: tokenizer_params['max_length'] = max_length
    def preprocess_function(examples):
        return tokenizer(examples['body'], **tokenizer_params)
    # tokenize dataset
    tokenized_dset = dset.map(preprocess_function, batched=True)
    print_gpu_utilization('dataset')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # load model and init trainer
    model = AutoModelForSequenceClassification.from_pretrained(model_label, num_labels=num_classes).to('cuda')
    print_gpu_utilization('model')
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy='epoch',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dset['train'],
        eval_dataset=tokenized_dset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

def eval_model(lang, model_folder, classes='all', rnd_seed=391772):
    if lang == 'en': df = corpus_telegram_en_ph1()
    elif lang == 'es': df = corpus_telegram_es_ph1()
    else: raise ValueError('supported languages are "en" and "es"')
    df = add_class_labels(df)
    if classes != 'all':
        df = binary_dataset(df, classes)
        score_fns = classif_scores('binary')
    else: score_fns = classif_scores('multiclass')
    df['label'] = df[CLS_TRNRY_IX]
    train, dev = train_test_split(df, test_size=0.2, random_state=rnd_seed, stratify=df[[CLS_TRNRY_IX]])
    dset = DatasetDict({'train': datasets.Dataset.from_pandas(train), 'test': datasets.Dataset.from_pandas(dev)})
    # load model
    print_gpu_utilization('start')
    device = torch.device('cuda:0')
    model = AutoModelForSequenceClassification.from_pretrained(model_folder).to(device)
    print_gpu_utilization('model')
    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device,
                                      max_length=256, truncation=True, batch_size=64)
    result = pipe(dset['test']['body'], function_to_apply='softmax')
    predicted = [int(r['label'][-1]) for r in result]
    for fn_label, fn in score_fns.items():
        res = fn(dev[CLS_TRNRY_IX], predicted)
        print(f'{fn_label}: {res:.3f}')

if __name__ == '__main__':
    # ENGLISH BINARY
    #build_model(lang='en', classes=[LBL_CRITIC, LBL_CONSP], model_label='bert-base-uncased', max_length=256, epochs=10)
    #eval_model(lang='en', classes=[LBL_CRITIC, LBL_CONSP], model_folder='results/checkpoint-1060')
    # SPANISH BINARY
    #build_model(lang='es', classes=[LBL_CRITIC, LBL_CONSP], model_label='dccuchile/bert-base-spanish-wwm-uncased', max_length=256, epochs=10)
    eval_model(lang='es', classes=[LBL_CRITIC, LBL_CONSP], model_folder='results/checkpoint-1160')
