from copy import copy
from typing import Union, Dict, Iterable
import regex as re

from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, Span, Token

from data_tools.seqlab_data.data_struct import SpanAnnotation


def map_annotation_to_spacy_tokens(doc: Doc, annotation: SpanAnnotation) -> Union[Span, None]:
    """
    Maps a SpanAnnotation to a range of tokens in a SpaCy Doc object.

    Args:
        doc: SpaCy Doc object containing the text.
        annotation: SpanAnnotation object representing the annotated span of text.

    Returns:
        A SpaCy Span object representing the range of tokens corresponding to the annotation,
        or None if there is no matching span in the doc.
    """

    # Ensure the annotation start and end are within the doc boundaries
    if annotation.start < 0 or annotation.end > len(doc.text):
        print(
            f"Error: Annotation boundaries {annotation.start}-{annotation.end} are outside the doc length {len(doc.text)}")
        return None

    # Find the start and end tokens corresponding to the annotation start and end
    start_token = None
    end_token = None
    for token in doc:
        if token.idx == annotation.start:
            start_token = token.i
        if token.idx + len(token) == annotation.end:
            end_token = token.i

    # Ensure start_token and end_token were found in the doc
    if start_token is None or end_token is None:
        print(
            f"Error: Couldn't find tokens corresponding to annotation boundaries {annotation.start}-{annotation.end} in the doc")
        return None

    # Return the span of tokens corresponding to the annotation
    return doc[start_token: end_token + 1]


from typing import List, Union


def map_annotations_to_spacy_tokens(doc: Doc, annotations: List[SpanAnnotation]) -> List[Union[Span, None]]:
    """
    Maps a list of SpanAnnotations to corresponding ranges of tokens in a SpaCy Doc object.

    Args:
        doc: SpaCy Doc object containing the text.
        annotations: List of SpanAnnotation objects representing the annotated spans of text.

    Returns:
        A list of SpaCy Span objects representing the ranges of tokens corresponding to the annotations,
        or None if there is no matching span in the doc for a given annotation.
    """

    # Sort the annotations by start index
    annotations.sort(key=lambda annotation: annotation.start)

    # Initialize the current token index
    current_token_index = 0

    # Initialize the list of spans
    spans = []

    # Iterate through the sorted annotations
    for annotation in annotations:
        # Ensure the annotation start and end are within the doc boundaries
        if annotation.start < 0 or annotation.end > len(doc.text):
            print(
                f"Error: Annotation boundaries {annotation.start}-{annotation.end} are outside the doc length {len(doc.text)}")
            spans.append(None)
            continue

        # Find the start and end tokens corresponding to the annotation start and end
        start_token = None
        end_token = None
        for token_index in range(current_token_index, len(doc)):
            token = doc[token_index]
            if token.idx == annotation.start:
                start_token = token.i
            if token.idx + len(token) == annotation.end:
                end_token = token.i
                current_token_index = token_index  # Update the current token index
                break  # Break the loop once the end token is found

        # Ensure start_token and end_token were found in the doc
        if start_token is None or end_token is None:
            print(
                f"Error: Couldn't find tokens corresponding to annotation boundaries {annotation.start}-{annotation.end} in the doc")
            spans.append(None)
            continue

        # Append the span of tokens corresponding to the annotation to the list of spans
        spans.append(doc[start_token: end_token + 1])

    return spans

def spacy_tok_check(doc, original_text):
    concatenated_tokens = '|'.join(token.text_with_ws for token in doc)
    if original_text != concatenated_tokens:
        print('Original text: ', original_text)
        print('Concatenated tokens: ', concatenated_tokens)
        assert original_text == concatenated_tokens, "Mismatch between original text and concatenation of tokens"

def check_tokenization_custom(doc: Doc, original_text: str) -> bool:
    '''
    Check if spacy Doc derived from original text matches the original text,
    by ignoring whitespaces and matching tokens to their corresponding text slices.
    '''
    text_idx = 0  # Start from the beginning of the original text
    for token in doc:
        if not token.text.isspace(): # ignore wspace tokens
            # Try to match the token text with a slice of the original text
            if token.text != original_text[text_idx:text_idx+len(token.text)]:
                # If they don't match, print the details and return False
                print(f"Tokenization mismatch at index {text_idx} in the text.")
                print(f"Original text: '{original_text}'")
                print(f"Expected token: '{original_text[text_idx:text_idx+len(token.text)]}'")
                print(f"Actual token: '{token.text}'")
                return False
            text_idx += len(token.text)  # Move the index to the end of the current token in the original text
        while text_idx < len(original_text) and original_text[text_idx].isspace(): # skip spaces in original text
            text_idx += 1

    # If there are extra characters in the original text after all tokens have been checked, print the details
    if text_idx != len(original_text):
        print(f"Extra characters found in the original text starting at index {text_idx}: '{original_text[text_idx:]}'")
        return False

    # If no issues were found, return True
    return True

def match_span_annots_to_spacy_spans_old(doc, annotations: List[SpanAnnotation]) -> Dict[SpanAnnotation, Span]:
    matched_spans = {}
    annotations = sorted(annotations, key=lambda ann: ann.start)
    unnmatched_annotations = []
    for ann in annotations:
        if ann.is_none:
            matched_spans[ann] = doc[0:0]
            continue
        # canonize spaces in the annotation text
        annotation_text = str(ann)
        annotation_text = re.sub(r'\s+', ' ', annotation_text)
        matched = False
        for i, token in enumerate(doc):
            if token.text.isspace(): continue
            for j, token2 in enumerate(doc[i:]):
                if token2.text.isspace(): continue
                token_seq = doc[token.i: token2.i+1]
                tokens_text = re.sub(r'\s+', ' ', token_seq.text)
                if tokens_text == annotation_text:
                    matched_spans[ann] = token_seq
                    matched = True
                    break
                if len(tokens_text) > len(annotation_text): break
            if matched: break
        if not matched:
            unnmatched_annotations.append(ann)
    # [f'[{str(t)}]:[{t.idx}-{t.idx+len(t)-1}]' for t in doc] tokens and their orig. spans
    # in spaCy, Doc.text is expected to correspond to the original text
    # here, this is not the case - probably because of the way the tokenization is done
    if unnmatched_annotations:
        for ann in unnmatched_annotations:
            # canonize spaces in the annotation text
            annotation_text = ann.text[ann.start: ann.end]
            annotation_text = re.sub(r'\s+', ' ', annotation_text)
            matched = False
            for i, token in enumerate(doc):
                if token.text.isspace(): continue
                for j, token2 in enumerate(doc[i:]):
                    if token2.text.isspace(): continue
                    token_seq = doc[token.i: token2.i + 1]
                    tokens_text = re.sub(r'\s+', ' ', token_seq.text)
                    if tokens_text == annotation_text:
                        matched_spans[ann] = token_seq
                        matched = True
                        break
                    if len(tokens_text) > len(annotation_text): break
                if matched: break
    return matched_spans


def annotation_matches_token_range(doc: Doc, ann: SpanAnnotation, tokeni: Token, tokenj: Token) -> bool:
    '''
    True if text of the span annotation matches the spacy token range, disregarding types of whitespaces between words.
    '''
    annotation_text = str(ann)
    annotation_text = re.sub(r'\s+', '', annotation_text)
    token_seq = doc[tokeni.i: tokenj.i + 1]
    tokens_text = re.sub(r'\s+', '', token_seq.text)
    return tokens_text == annotation_text

def deepcopy_datastruct(data):
    '''
    Copy recursively, but only lists, dicts, and sets. For other objects, just re-use references.
    :return: the copy
    '''
    if isinstance(data, list):
        return [deepcopy_datastruct(item) for item in data]
    elif isinstance(data, dict):
        return {deepcopy_datastruct(key): deepcopy_datastruct(value) for key, value in data.items()}
    elif isinstance(data, set):
        return {deepcopy_datastruct(item) for item in data}
    else:
        return data

def match_span_annots_to_spacy_spans(doc, annotations: List[SpanAnnotation], debug=False) -> Dict[SpanAnnotation, Span]:
    #matched_spans = { ann: [] for ann in annotations }
    matched_spans = {}
    unnmatched_annotations = []
    # for each none annotation, match with a span of length 0, then remove it from the list
    for ann in annotations:
        if ann.is_none: matched_spans[ann] = doc[0:0]
    annotations = [ann for ann in annotations if not ann.is_none]
    # for each ann, potential_spans must be sorted by token position
    potential_start_spans = {}
    potential_end_spans = {}
    def add_potential_span(pmap, ann, token):
        if ann not in pmap: pmap[ann] = []
        pmap[ann].append(token)
    for ann in annotations:
        annotation_text = str(ann)
        #annotation_text = re.sub(r'\s+', ' ', annotation_text) # canonize spaces in the annotation text
        for i, token in enumerate(doc): # potential spans will be sorted cause tokens are iterated in order
            if token.text.isspace(): continue
            if annotation_text.startswith(token.text): add_potential_span(potential_start_spans, ann, token)
            if annotation_text.endswith(token.text): add_potential_span(potential_end_spans, ann, token)
        if ann not in potential_start_spans or ann not in potential_end_spans:
            if ann not in potential_start_spans:
                print(f'No potential start span found for annotation [{str(ann)}]')
                potential_start_spans[ann] = []
            if ann not in potential_end_spans:
                print(f'No potential end span found for annotation [{str(ann)}]')
                potential_end_spans[ann] = []
            # print tokens in the doc, in one line, separated by ;
            print(';'.join(t.text for t in doc))
        potential_start_spans[ann] = sorted(potential_start_spans[ann], key=lambda token: token.i)
        potential_end_spans[ann] = sorted(potential_end_spans[ann], key=lambda token: token.i)
    # process annots for a single author at a time, then among these annots with the same text
    # this is to avoid overlapping annots whan scanning the spacy spans to match
    # for diff. author or text, potentially annotations can overlap (in spans)
    ann_by_author = group_by_perm(annotations, lambda ann: ann.author)
    for author, ann_list_auth in ann_by_author.items():
        ann_by_text = group_by_perm(ann_list_auth, lambda ann: str(ann)) # group the annotation by their text
        for ann_text, ann_list_txt in ann_by_text.items():
            # sort the annotations by their start
            anns = sorted(ann_list_txt, key=lambda ann: ann.start)
            start_spans, end_spans = deepcopy_datastruct(potential_start_spans), deepcopy_datastruct(potential_end_spans)
            # todo: handle the case where there is a lot of (same text) annotation, with missing annotations in between
            #      this algorithm scans the tokens in order, so the annotations will be matches sequentially,
            #      but in these case the missing annotation gap will not be respected, ie, spans will be shifted to the left
            #      however, this should be a rare case
            for ann in anns:
                annx = str(ann)
                matched = False
                next_ann = False
                for tokeni in start_spans[ann]:
                    for tokenj in end_spans[ann]:
                        if tokenj.i < tokeni.i: continue
                        if annotation_matches_token_range(doc, ann, tokeni, tokenj):
                            matched_spans[ann] = doc[tokeni.i: tokenj.i + 1]
                            matched = True
                            if debug: print('MATCH!: ' + str(ann))
                            # remove token form start and end spans for all other anns
                            for ann2 in anns:
                                if ann2 == ann: continue
                                if tokeni in start_spans[ann2]:
                                    start_spans[ann2] = [t for t in start_spans[ann2] if t.i != tokeni.i]
                                if tokenj in end_spans[ann2]:
                                    end_spans[ann2] = [t for t in end_spans[ann2] if t.i != tokenj.i]
                                if debug and str(ann2) in []: # detailed data for target spans, for debug
                                    # print start and end spans, each in one line, separated by ;
                                    print(';'.join(t.text for t in start_spans[ann2]))
                                    print(';'.join(t.text for t in end_spans[ann2]))
                            next_ann = True
                        if next_ann: break
                    if next_ann: break
                if not matched:
                    unnmatched_annotations.append(ann)
    return matched_spans

def group_by_perm(data, keyf):
    ''' group_by in a permanent re-iterable data structure with the same interface, with data copied '''
    keys = set([keyf(d) for d in data])
    res = { k: [] for k in keys }
    for d in data: res[keyf(d)].append(copy(d))
    for k, itms in res.items():
        keys = set([keyf(it) for it in itms])
        assert len(keys) == 1
        assert keys.pop() == k
    return res

def labelset_to_string(labels: Iterable[str]) -> str:
    labels = list(set(labels))
    labels = sorted(labels)
    return ';'.join(labels)

LABELSET_RESOLUTION_RULES = {
	'A;E' : 'A' ,
	'A;F' : 'F' ,
	'A;O' : 'A' ,
	'E;O' : 'O' ,
	'E;P' : 'P' ,
	'E;P;V' : 'P' ,
	'E;V' : 'V' ,
	'F;O' : 'F' ,
	'F;V' : 'F' ,
	'O;P' : 'P' ,
	'P;V' : 'P' ,
    'O;V' : 'V' ,
    'F;P;V': '?',
    'A;P': '?',
    'F;P': '?',
    'A;V': '?',
}

def resolve_label(labels: List[str]) -> str:
    '''
    For a case when the same span has multiple labels, resolve this label set to a single label
    :param labels:
    :return:
    '''
    if len(labels) == 1: return labels[0]
    if len(set(labels)) == 1: # there is only one label, but duplicated
        res = labels[0]
    else:
        lset = labelset_to_string(labels)
        res = LABELSET_RESOLUTION_RULES.get(lset, '?')
        if res == '?':
            print('WARNING: cannot resolve labelset: ' + lset)
            res = labels[0]
    return res

def process_duplicate_span_labels(spans: List[SpanAnnotation], output_labelsets=False, verbose=False, stats=False) -> List[SpanAnnotation]:
    '''
    Group spans by text_id and remove the duplicates, which can occur due to errors.
    :return: list of spans with no duplicates, where the duplicates are removed according to a set of rules
    '''
    # group spans by text_id
    labelsets = set()
    labelset_cases = {}
    spans_by_text_id = group_by_perm(spans, lambda span: span.text_id)
    # remove duplicates
    new_spans = []
    for text_id, spans in spans_by_text_id.items():
        # check if there are two spans, not none, with exact same boundaries but different labels
        spans_by_boundaries = group_by_perm(spans, lambda span: (span.start, span.end, span.author))
        for (start, end, author), spans in spans_by_boundaries.items():
            if len(spans) > 1: # duplicate spans - more than one span with same boundaries and author
                #sptxt = spans[0].text
                labels = [span.label for span in spans]
                lss = labelset_to_string(labels)
                if len(set(labels)) > 1: labelsets.add(lss)
                resolved_label = resolve_label(labels)
                if verbose and lss not in LABELSET_RESOLUTION_RULES:
                    num, stxt = spans[0].text_num, str(spans[0])
                    print(f'Warning: duplicate spans with different labels found for text_num {num} text_id {text_id}\nauthor {author}, '
                          f'span text: {stxt}, start {start}, end {end}, labels {labels}\n')
                text = spans[0].text
                new_spans.append(SpanAnnotation(text_id=text_id, text=text, start=start,
                                                end=end, label=resolved_label, author=author))
                if lss not in labelset_cases: labelset_cases[lss] = []
                labelset_cases[lss].append((text_id, author))
            else:
                new_spans.append(spans[0])
    if output_labelsets:
        labelsets = sorted(list(labelsets))
        print('\nLABELSET_RESOLUTION_RULES = {')
        for ls in labelsets:
            print(f"\t'{ls}' : '?' , ")
        print('}\n')
    print()
    if stats:
        print(f'Labelsets with multiple labels:')
        for ls, cases in labelset_cases.items():
            if len(ls) == 1: continue # single label (that was duplicated), this is correctable error
            print(f'Labelset {ls} has {len(cases)} occurrences in {len(set([c[0] for c in cases]))} texts')
        print()
        for ls, cases in labelset_cases.items():
            if (ls.endswith(';V') and len(ls) == 3) or len(ls) == 1: continue # these are expected duplicates, skip
            print(f'Instances of labelset {ls} ({len(set([c[0] for c in cases]))} cases):')
            print(f'{";".join([f"{c[0]}:{c[1]}" for c in cases])}')
            print()
    return new_spans
