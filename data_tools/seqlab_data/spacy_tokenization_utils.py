import spacy
from spacy.attrs import ORTH


def add_splitting_special_tok_cases(nlp):
    # define cases when a string should be split in two: (string, split after char index)
    cases = [('):', 1)]
    for s, ci in cases:
        special_case = [{ORTH: s[:ci]}, {ORTH: s[ci:]}]
        nlp.tokenizer.add_special_case(s, special_case)

def customize_spacy_tokenizer(nlp):
    # TODO: '\u200C\u200B\u200C' is rare for now, these are 'zero-width' chars
    # TODO: if more cases occur, do a 'whitespace normalization' step before tokenization
    prefixes = nlp.Defaults.prefixes + ['•', '-', '~', '\u200C\u200B\u200C', '‘’', r'\[\*', r'\u200C\u200C', r'\(',
                                        r'\[', '"']
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search
    infixes = nlp.Defaults.infixes + ['—', '\|', '>>', '\+', '=>', r'(?<=[IVXLCDM])-(?=[a-zA-Z])',
                                      '(?<=[0-9])\.(?=[a-zA-Z])', '-', '!!!']
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    suffixes = nlp.Defaults.suffixes + ['/', ':', '%', '\u200C\u200B\u200C', '\u200C\u200C', '’’', r'\)\]', r'\]', '@',
                                        '-', r'\)', '"']
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    add_splitting_special_tok_cases(nlp)
