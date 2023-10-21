from dataclasses import dataclass
from typing import List

from data_tools.seqlab_data.span_data_definitions import NONE_LABEL


@dataclass
class AuthLabelTxtSpans():
    '''
    List of spans for a single text, single author and single label.
    '''
    label: str
    text: str
    text_id: str
    author: str
    spans: List

@dataclass
class SpanAnnotation():
    text_id: str
    text: str
    label: str
    author: str
    text_num: str = None
    start: int = None
    end: int = None
    # indices of the span tokens in the spacy doc
    spacy_start: int = None
    spacy_end: int = None

    def __eq__(self, other):
        return self.text_id == other.text_id and self.text == other.text and \
               self.label == other.label and self.author == other.author and \
               self.start == other.start and self.end == other.end

    def __len__(self):
        return self.end - self.start + 1

    def __hash__(self):
        return hash((self.text_id, self.text, self.label, self.author, self.start, self.end))

    def __str__(self):
        if self.is_none: return f'NONE'
        else: return self.text[self.start:self.end+1]

    @property
    def is_none(self):
        return self.label == NONE_LABEL


