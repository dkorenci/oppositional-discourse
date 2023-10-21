import json, re
import math
import traceback
from copy import copy
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Tuple

from data_tools.seqlab_data.data_struct import SpanAnnotation
from data_tools.seqlab_data.seqlabel_data_old import parse_json_annotations
from data_tools.seqlab_data.span_data_definitions import NONE_LABEL, LABEL_DEF, CONSPI_FIELDS, CT_FIELDS
from settings import AUTOGOLD_ANNOTS_EN_RAW, AUTOGOLD_ANNOTS_ES_RAW


def is_camel_case(s):
    if not s.isalpha(): return False
    # Check if string contains an uppercase letter that is not the first character
    if not re.search(r'.*[A-Z]', s[1:]): return False
    if s.isupper(): return False
    camel_case = re.match(r'^(?:(?:[a-z]+[A-Z])|[A-Z][a-z]*)(?:[A-Za-z]*)$', s)
    return camel_case is not None

def align_start_end(correct_chunk: str, text: str, start: int, end: int) -> Tuple[int, int]:
    '''
    Correct start and end indices of a text so that text[start:end+1] == correct_chunk
    :return: corrected start and end
    '''
    # iterate from start to end and find first match with correct_chunk
    for i in range(start, end+1):
        if text[i:i+len(correct_chunk)] == correct_chunk:
            return i, i+len(correct_chunk)-1
        if i+len(correct_chunk) > len(text): break
    return None, None


def hanging_letter(off_text_glob: str) -> bool:
    ''' If the is a whitespace-separated letter at the start -1, or at the end of the text 1, 0 otherwise.'''
    parts = off_text_glob.split()
    if len(parts) == 0: return 0
    # if string starts or ends with a space, this will be dealt with later by SpanCorrector
    if len(parts[0]) == 1 and not off_text_glob[0].isspace(): return -1
    if len(parts[-1]) == 1 and not off_text_glob[-1].isspace(): return 1
    return False

class LabelStudioParser():
    '''
    Parses json files from Label Studio
    '''

    def __init__(self, label_def : Dict[str, str] = LABEL_DEF, none_label =NONE_LABEL,
                 conspi_fields =CONSPI_FIELDS, ct_fields =CT_FIELDS, verbose = False, debug=False):
        '''
        :param label_def: label definitions, dictionary: label name -> label abbreviation
            must be in line with the labels in the json files
        '''
        self.label_def = label_def
        self.none_label = none_label
        self.conspi_fields = conspi_fields
        self.ct_fields = ct_fields
        self.verbose = verbose
        self.debug = debug

    def parse_folder(self, folder):
        '''
        Parse all .json annotation files in a folder, unify and complete the annotations.
        :param folder:
        :return:
        '''
        if self.verbose:
            print('Analyzing folder: ', folder)
            print('Parsing the files ...')
        annots = self._parse_folder_raw(folder)
        if len(annots) == 0: raise ValueError(f'No annotations found in folder {folder}')
        all_authors = set(a.author for a in annots)
        file_select = lambda x: x.text_id
        annots.sort(key=file_select)
        def group_annots(annots, select):
            res = []
            for k, g in groupby(annots, select):
                res.append((k, list(g)))
            return res
        annots_byfile = group_annots(annots, file_select)
        self._check_file_nums(annots_byfile)
        fid2txt = {}
        txt_mismatch_count = 0
        for ann in annots:
            fid, txt = ann.text_id, ann.text
            if fid not in fid2txt:
                fid2txt[fid] = txt
            else:
                if txt != fid2txt[fid]:
                    txt_mismatch_count += 1
                    print (f'WARNING: file id {fid} occurs more than once with different texts')
                    print(f'[{txt}]')
                    print(f'[{fid2txt[fid]}]')
                    print()
                    assert len(txt) == len(fid2txt[fid])
                    #raise ValueError(f'File id {fid} occurs more than once with different texts')
                #assert txt == fid2txt[fid]
        if txt_mismatch_count:
            print(f'!!!!!!! WARNING Number of texts with mismatching texts: {txt_mismatch_count}')
        result = []
        for file_id, annots in annots_byfile:
            try:
                ann_list = [ann for ann in annots]
                complete_annots = self._complete_annotations(ann_list, all_authors=all_authors, file_id=file_id,
                                                             text=fid2txt[file_id])
                result.extend(complete_annots)
            except Exception as e:
                etxt = f'ERROR for file_id: {file_id} - {e}'
                print(etxt)
                print(traceback.format_exc())
        return result

    def _check_file_nums(self, annots_byfile):
        ''' Check if the same file_id texts have the same file_num. '''
        numsg = set()
        if self.verbose: print(f'Number of texts: {len(annots_byfile)}')
        for file_id, annots in annots_byfile:
            numsl = set(an.text_num for an in annots)
            if len(numsl) != 1:
                raise ValueError(f'for text {file_id} different file_nums occurr across annotators: {numsl}')
            file_num = list(numsl)[0]
            if file_num in numsg:
                raise ValueError(f'file_num {file_num} occurrs for more than one file_id')
            else:
                numsg.add(file_num)

    def _complete_annotations(self, annots, all_authors, file_id, text):
        complete = []
        authors_none = []
        num_annot = 0
        authors = []
        for ann in annots:
            annotator, label = ann.author, ann.label
            assert (ann.text_id == file_id)
            if label != self.none_label:  # standard annotation
                if label in self.label_def:
                    label = self.label_def[label]
                else:
                    raise ValueError(f'NO ABBREVIATION FOR LABEL: {label} IN TEXT {ann.text_id}')
                annc = copy(ann)
                annc.label = label
                complete.append(annc)
                authors.append(annotator)
                num_annot += 1
            else:  # special label indicating that the annotator has no annotations for the file
                authors_none.append(annotator)
        if num_annot == 0:
            if self.verbose: print('WARNING: no annotations in the file')
        assert len(authors_none) == len(set(authors_none))  # check that each 'none' author occurs only once
        authors_none = set(authors_none)
        authors = set(authors)
        # authors that haven't got the file_id in their .json
        # so they aren't present but labeled with NONE_LABEL
        authors_notinfile = set(all_authors).difference(authors.union(authors_none))
        all_empty = authors_none.union(authors_notinfile)
        if len(all_empty):
            if self.verbose:
                print(f'Warning: "NONE" author occurr: {authors_none.union(authors_notinfile)}')
                print(f'    added authors: {authors}')
            for a in all_empty:
                complete.append(SpanAnnotation(text_id=file_id, text=text, label=self.none_label, author=a))
                #complete.append({'file_id': file_id, 'text': text, 'label': self.none_label, 'author': a})
        return complete

    def _parse_folder_raw(self, folder, print_labels=True) -> List[SpanAnnotation] :
        '''
        Parse annotation files in a folder.
        '''
        annots = []
        unique_labels = lambda annots: sorted(set(ann.label for ann in annots))
        for pth in Path(folder).glob('*.json'):
            if self.verbose: print(pth)
            file_annots = self.parse_json_file(pth)
            if self.verbose: print('ALL FILE LABELS: ', unique_labels(file_annots))
            annots.extend(file_annots)
        if print_labels and self.verbose: print('ALL LABELS:', unique_labels(annots))
        return annots

    def parse_json_file(self, pth) -> List[SpanAnnotation] :
        '''
        Check filename of a json file with Label Studio annotations, and parse it.
        Asumes one annotator per file, and annotator name embedded in file name.
        '''
        try:
            annotator_name = pth.name.split('.')[-2].split('_')[-1]
        except:
            raise Exception(f'Error, file name must end with _AUTHORNAME.json. File name: {pth.name}')
        parse = json.load(open(pth, 'r'))
        annots = self._extract_annot(parse, annotator_name)
        return annots

    def _extract_annot(self, jprs, annotator) -> List[SpanAnnotation] :
        '''
        Extract annotations from json data structure with annotations created using Labels Studio.
        Assume each annotation correspond to one text document, annotated by one annotator.
        '''
        result = []
        def get_label_field(d, fields):
            ''' Extract the value of one of the fields from d, as int. '''
            for f in fields:
                if f in d: return int(d[f])
            return None
        for r in jprs:
            # print(r['file_upload'], r['data']['id'])
            data = r['data']
            file_id, text, file_num = data['id'], data['body'], int(data['NUM'])
            # ignore text if it contains no critical thinking and no conspiracy theory
            conspi = get_label_field(data, self.conspi_fields)
            critic = get_label_field(data, self.ct_fields)
            if conspi is not None and critic is not None:
                if conspi == 0 and critic == 0:
                    if self.verbose: print(f'WARNING: 0-0 labels for text {file_id}, ignoring')
                    continue
            else:
                if self.verbose: print(f'Nonexistent or partial GS metadata in {file_id}')
            annots = r['annotations']
            if len(annots) != 1:
                if self.verbose:
                    print(f'WARNING: length of "annotations" block is not 1 but {len(annots)}\nANNOTS:{annots}')
            spans = annots[0]['result']
            if len(spans) == 0:  # no spans at all for this annotator, add special annotation
                if self.verbose:
                    print(f'NO SPANS for the file {file_id}, annotator: {annotator}')
                result.append(SpanAnnotation(text_id=file_num, text_num=file_num, text=text,
                                             author=annotator, label=self.none_label))
            else:
                startend_mismatch = False;
                empty_span = False
                for span in spans:
                    # read span data
                    assert span['type'] == 'hypertextlabels'
                    spdata = span['value']
                    start, end = spdata['startOffset'], spdata['endOffset']
                    start_global, end_global = spdata['globalOffsets']['start'], spdata['globalOffsets']['end']
                    # check for span boundaries errors and warnings
                    if start != start_global or end != end_global: startend_mismatch = True
                    if start_global == end_global:
                        empty_span = True
                        continue  # ignore empty spans
                    elif start_global > end_global:
                        raise ValueError(f'Span start ({start_global}) > end ({end_global}) in file {file_id}')
                    # get label data and create an annotation record
                    labels = spdata['hypertextlabels']
                    assert len(labels) == 1
                    label = labels[0]
                    if end_global >= len(text): end_global = len(text) - 1 # correct out-of-txt-boundary error
                    # default span, without any corrections
                    default_span = SpanAnnotation(text_id=file_num, text_num=file_num, text=text,
                                                 author=annotator, label=label, start=start_global, end=end_global)
                    # check if the text matched the offset-derived text
                    span_text = spdata['text'] if 'text' in spdata else None
                    off_text_glob, off_text = text[start_global: end_global+1], text[start: end+1]
                    # detect and correct special cases due to label studio errros
                    if span_text is not None and span_text != off_text_glob:
                        span_start = text.find(span_text)
                        start_correct = None
                        match_correct = False
                        if is_camel_case(off_text_glob) and \
                        (off_text_glob.startswith(span_text) or off_text_glob.endswith(span_text)):
                            # probably this is a wrongly included char in off_text_glob, span_text should be correct
                            start_correct, end_correct = align_start_end(span_text, text, start_global, end_global)
                        # 'hanging letter' at start or end
                        elif hanging_letter(off_text_glob) == -1 and off_text_glob.endswith(span_text) or \
                                hanging_letter(off_text_glob) == 1 and off_text_glob.startswith(span_text):
                            start_correct, end_correct = align_start_end(span_text, text, start_global, end_global)
                        elif span_start != -1 and text.find(span_text, span_start + 1) == -1: # span_text is in text, only once
                            start_correct, end_correct = align_start_end(span_text, text, 0, len(text) - 1)
                        # if off_text_glob contains whitespace and span_text is not its prefix or suffix
                        elif re.search(r'\s', off_text_glob) \
                            and not off_text_glob.startswith(span_text) and not off_text_glob.endswith(span_text):
                            # this is a 'degenerate text' case, probably due to label studio error
                            start_correct, end_correct = align_start_end(span_text, text, start_global, end_global)
                        # if start_correct is not None (re-alignment was performed), create new span with corrected offsets
                        if start_correct is not None:
                            result.append(SpanAnnotation(text_id=file_num, text_num=file_num, text=text,
                                                         author=annotator, label=label, start=start_correct, end=end_correct))
                            if self.debug:
                                print(f'Corrected offsets in text {file_id}, label {label}, annotator {annotator}')
                                print(f'Global offsets: {start_global:10}, {end_global:10}, text: {text[start_global:end_global+1]}')
                                print(f'Corrected offs: {start_correct:10}, {end_correct:10}, text: {text[start_correct:end_correct+1]}')
                                print()
                            continue
                        else:
                            result.append(default_span)
                    else: result.append(default_span)
                    strip_chars = ' \n\t,.;:!?)’(‘"“”-—–[]{}<>«»/-@…\'`_#¿¡'
                    off_text_glob, off_text = off_text_glob.strip(strip_chars), off_text.strip(strip_chars)
                    span_text = span_text.strip(strip_chars) if span_text is not None else None
                    if self.debug:
                        if span_text is None or span_text.strip() != off_text_glob.strip():
                            # texts differ, but exclude mostly bening cases of one char at start/end mismatch
                            if (span_text.startswith(off_text_glob) or span_text.endswith(off_text_glob) \
                                or off_text_glob.startswith(span_text) or off_text_glob.endswith(span_text))\
                                and math.fabs(len(off_text_glob)-len(span_text)) == 1:
                                pass
                            elif self.debug:
                                print(f'TEXT span text and offset-derived text mismatch in file {file_id}, author {annotator}: ')
                                print(f' span text: [{span_text}]\noffset-txt: [{off_text_glob}]\n')
                        if empty_span:
                            print(f'WARNING, start == end for a span in text {file_id}')
        return result

def ls_parse_all_subfolders(folder, ls_parser=None, verbose=False, debug=False) -> List[SpanAnnotation]:
    '''
    Parse all the label studio annotations in the subfolders of the folder.
    '''
    if ls_parser is None: ls_parser = LabelStudioParser(verbose=verbose, debug=debug)
    return [ann for f in Path(folder).glob('*') for ann in ls_parser.parse_folder(f)]

def test_against_the_old_code(folder):
    '''
    Test the new parser against the old one, by parsing all the annotations in the folder
    with both parsers and comparing the results.
    '''
    parse_new = ls_parse_all_subfolders(folder)
    parse_old = parse_json_annotations(folder)
    # function for converting all dict-style annotation to SpanAnnotation object
    def convert_to_spanannot(ann):
        return SpanAnnotation(text_id=ann['file_id'], text=ann['text'], author=ann['author'], label=ann['label'],
                              text_num=ann['file_num'] if 'file_num' in ann else None,
                              start=ann['start'] if 'start' in ann else None,
                              end=ann['end'] if 'end' in ann else None)
    for i, (a, b) in enumerate(zip(parse_new, parse_old)):
        if a != convert_to_spanannot(b):
            print(f'ERROR: annotation {i} is different')
            print('NEW:', a)
            print('OLD:', convert_to_spanannot(b))
            print()

def parse_analysis(lang='en'):
    if lang == 'en': dfolder = AUTOGOLD_ANNOTS_EN_RAW
    else: dfolder = AUTOGOLD_ANNOTS_ES_RAW
    spans = ls_parse_all_subfolders(dfolder, debug=True)

def test_camel_case():
    print(is_camel_case('hello'))  # False
    print(is_camel_case('Hello'))  # False
    print(is_camel_case('helloWorld'))  # True: This is a valid lowerCamelCase string
    print(is_camel_case('HelloWorld'))  # True: This is a valid UpperCamelCase string
    print(is_camel_case('hello_world'))  # False: This uses underscore, not CamelCase
    print(is_camel_case('Hello_world'))  # False: This uses underscore, not CamelCase
    print(is_camel_case('helloWorld123'))  # False: This string contains digits

if __name__ == '__main__':
    #test_against_the_old_code(SEQLABEL_ANNOTS_EN_RAW)
    #test_against_the_old_code(SEQLABEL_ANNOTS_ES_RAW)
    parse_analysis('en')
    #test_camel_case()