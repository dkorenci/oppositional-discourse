from typing import List
from spacy.tokens import DocBin, Doc
import pickle

from data_tools.classif_data.data_loaders import classif_corpus_raw_df, CLASSIF_DF_ID_COLUMN
from data_tools.classif_data.data_utils import binary_dataset, LBL_CONSP, LBL_CRITIC, CLS_BINARY_IX, add_class_labels
from data_tools.seqlab_data.seqlab_utils import group_by_perm, annotation_matches_token_range
from data_tools.seqlab_data.create_spacy_span_dataset import convert_spans_to_spacy
from data_tools.seqlab_data.span_data_loader import load_annotations
from data_tools.seqlab_data.span_data_definitions import NONE_LABEL
from data_tools.spacy_utils import ON_DOC_ID, create_spacy_model, define_spacy_extensions, \
    ON_DOC_CLS_EXTENSION
from data_tools.seqlab_data.url_tokenization import SpacyURLTokenizer
from settings import SPACY_DSET_ES, SPACY_DSET_EN, SPACY_DSET_ALLSPANS_EN, SPACY_DSET_ALLSPANS_ES

define_spacy_extensions()

def spacy_dset_name(lang, extension):
    return f'dset_{lang}.{extension}'

def save_spacy_dataset_docbin(lang, dset: List[Doc]):
    spacy_dset = DocBin(docs=dset, store_user_data=True)
    spacy_dset.to_disk(spacy_dset_name(lang, extension='spacy'))

def load_spacy_dataset_docbin(lang, local=False, allspans=False) -> List[Doc]:
    if local: fname = spacy_dset_name(lang, extension='spacy') # version in the current working folder
    else: # official, fixed, version
        if allspans:
            fname = SPACY_DSET_ALLSPANS_EN if lang == 'en' else SPACY_DSET_ALLSPANS_ES
        else:
            fname = SPACY_DSET_EN if lang == 'en' else SPACY_DSET_ES
    doc_bin = DocBin().from_disk(fname)
    nlp = create_spacy_model(lang, fast=True)
    nlp.tokenizer = SpacyURLTokenizer(nlp)
    return list(doc_bin.get_docs(nlp.vocab))

def save_spacy_dataset_pickle(lang, dset: List[Doc]):
    with open(spacy_dset_name(lang, extension='pickle'), 'wb') as f:
        pickle.dump(dset, f)

def load_spacy_dataset_pickle(lang) -> List[Doc]:
    with open(spacy_dset_name(lang, extension='pickle'), 'rb') as f:
        return pickle.load(f)

def compare_and_report(doc, span_annotations):
    unmatched_spans = []
    unmatched_annotations = list(span_annotations)  # Start with all annotations unmatched
    none_label_found = False
    for label, start, end, author in doc._.opn_spans:
        if label == NONE_LABEL: none_label_found = True
    if none_label_found: assert len(doc._.opn_spans) == 1 # NONE label should be the only one!
    for label, start, end, author in doc._.opn_spans:
        span_text = doc[start:end].text
        match_found = False
        for sa in unmatched_annotations:
            match_found = False
            if label != NONE_LABEL: # regular span, match all properties
                if sa.label == label and sa.author == author and \
                        annotation_matches_token_range(doc, sa, doc[start], doc[end-1]):
                    match_found = True
            elif sa.label == NONE_LABEL and sa.author == author: match_found = True
            if match_found:
                unmatched_annotations.remove(sa)  # Remove the matched annotation
                break
        if not match_found:
            unmatched_spans.append((label, span_text, author))
    # Report the unmatched items
    if unmatched_spans or unmatched_annotations:
        if unmatched_spans:
            print("Unmatched doc spans:")
            for span in unmatched_spans:
                print(span)
            print()
            #raise ValueError(f'{len(unmatched_spans)} unmatched spans for doc id {doc._.get(ON_DOC_ID)}')
        #print("Unmatched SpanAnnotations:", unmatched_annotations, "\nNumber:", len(unmatched_annotations))
        return len(unmatched_annotations)
    return True # all spans and annotations matched

def test_spacy_dataset(lang):
    '''
    Compare deserialized DocBin dataset with the original span and classif. datasets.
    '''
    # load classif. data, creat class variable
    classif_df = classif_corpus_raw_df(lang); print('loaded classif. data')
    classif_df = binary_dataset(add_class_labels(classif_df), classes=[LBL_CONSP, LBL_CRITIC])
    id2class = {id: cls for id, cls in zip(classif_df[CLASSIF_DF_ID_COLUMN], classif_df[CLS_BINARY_IX])}
    # load raw span dataset
    spans = load_annotations(lang, gold=True, annotator_data=False); print('loaded span data')
    spans_by_id = group_by_perm(spans, lambda span: span.text_id)
    id2spans = {id: spans for id, spans in spans_by_id.items()}
    # load spacy dataset
    docs = load_spacy_dataset_docbin(lang);
    print(f'loaded spacy data: {len(docs)} docs')
    mismatch_count = 0; span_missmatch_count = 0; span_annot_count = 0
    for doc in docs:
        #print_doc_extensions(doc)
        #continue
        id = doc._.get(ON_DOC_ID)
        assert id in id2spans, f'id {id} not found in spans'
        assert id in id2class, f'id {id} not found in classif'
        assert doc._.get(ON_DOC_CLS_EXTENSION) == id2class[id], f'classif. label mismatch for id {id}'
        spans = id2spans[id]
        span_annot_count += len(spans)
        res = compare_and_report(doc, spans)
        if res != True:
            span_missmatch_count += res
            mismatch_count += 1
            #print(f'!Mismatch for id: {id}\n')
        #if len(spans) != len(doc._.get(ON_DOC_EXTENSION)):
            #f'span count mismatch for id {id}'
    print(f"Number of docs with a mismatch: {mismatch_count}, percentage: {mismatch_count/len(docs)*100:.3f}")
    print(f"Number of unmatched spans: {span_missmatch_count}, percentage: {span_missmatch_count/span_annot_count*100:.3f}")

def create_spacy_dataset(lang, save=False, test_only=False, verbose=False, annotator_data=False):
    '''
    Create a spacy dataset from the classif and span datasets, produce Doc objects with both span and class labels.
    :param lang: en or es
    :param save:
    :param test_only: False or an int (for testing) - number of texts to process
    :return:
    '''
    # load raw classif dataset, check ids
    classif_df = classif_corpus_raw_df(lang)
    classif_ids = classif_df[CLASSIF_DF_ID_COLUMN]
    assert len(classif_ids) == len(set(classif_ids)), 'ids must be unique'
    # load spacy docs with span annotations
    span_docs = convert_spans_to_spacy(lang, id_filter=list(classif_ids), gold=True, annotator_data=annotator_data,
                                       verbose=verbose, stop_after=test_only)
    # create binary classif. labels: conspiracy==0, critical==1
    classif_df = binary_dataset(add_class_labels(classif_df), classes=[LBL_CONSP, LBL_CRITIC])
    id2class = {id: cls for id, cls in zip(classif_df[CLASSIF_DF_ID_COLUMN], classif_df[CLS_BINARY_IX])}
    # set class labels on the docs
    for doc in span_docs: doc._.set(ON_DOC_CLS_EXTENSION, id2class[doc._.get(ON_DOC_ID)])
    if save: save_spacy_dataset_docbin(lang, span_docs)

if __name__ == '__main__':
    create_spacy_dataset('en', save=True, test_only=False, verbose=True, annotator_data=True)
    #test_spacy_dataset('en')
