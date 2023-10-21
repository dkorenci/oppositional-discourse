import spacy
from spacy.tokens import Doc, Span

from data_tools.seqlab_data.spacy_tokenization_utils import customize_spacy_tokenizer

DEFAULT_ES_MODEL = 'es_core_news_sm'
DEFAULT_EN_MODEL = 'en_core_web_sm'

# names of the extension properties for spacy docs and spans
ON_DOC_EXTENSION = 'opn_spans' # doc._.opn_spans
#ON_SPAN_EXTENSION = 'label' # span._.label
#ON_SPAN_AUTHOR = 'author' # span._.author
ON_DOC_ID = 'doc_id' # doc._.id

# names of the extension property for doc class label
ON_DOC_CLS_EXTENSION = 'opn_class' # doc._.opn_spans

__EXTENSIONS_DEFINED = False

def define_spacy_extensions():
    global __EXTENSIONS_DEFINED
    if __EXTENSIONS_DEFINED: return
    Doc.set_extension(ON_DOC_CLS_EXTENSION, default=None) # classif. label extension
    Doc.set_extension(ON_DOC_EXTENSION, default=[]) # span annotations
    Doc.set_extension(ON_DOC_ID, default=None) # doc id
    #Span.set_extension(ON_SPAN_EXTENSION, default=None)
    #Span.set_extension(ON_SPAN_AUTHOR, default=None)
    __EXTENSIONS_DEFINED = True

def print_doc_extensions(doc):
    # get all custom attributes from define_spacy_extensions() and print them
    for attr in dir(doc._):
        if attr.startswith('_'): continue
        print(f'{attr}: {getattr(doc._, attr)}')

def create_spacy_model(lang, fast=False):
    if lang == 'en': model = DEFAULT_EN_MODEL
    elif lang == 'es': model = DEFAULT_ES_MODEL
    else: raise ValueError(f'lang {lang} not supported')
    nlp = spacy.load(model) if not fast else spacy.load(model, disable=['ner', 'parser'])
    customize_spacy_tokenizer(nlp)
    return nlp
