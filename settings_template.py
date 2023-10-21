'''
definitions of important setting variables (resources etc.)
create a local settings.py (git ignored) with these variables and their appropriate values
when introducing new variables, add them to this file with empty or default values and commit
'''

# conspiracy vs critical thinking vs NONE
CORPUS_TELEGRAM_ES_PHASE1 = ''
CORPUS_TELEGRAM_EN_PHASE1 = ''

HEADLESS = False # turn on for server experimens to avoid graphic errors

# per-language folders with sequence-labeled texts, one sub-folder per batch
ANNOTS_EN_RAW = 'datafolder/json/EN/'
ANNOTS_ES_RAW = 'datafolder/json/ES/'

ANNOTS_EN_RAW_DEV = ''
ANNOTS_ES_RAW_DEV = ''

SEQLABEL_ANNOTS_ES_RAW = ''
SEQLABEL_ANNOTS_EN_RAW = ''

SEQLABEL_ANNOTS_ES_JSON = ''
SEQLABEL_ANNOTS_EN_JSON = ''

SEQLABEL_ANNOTS_EN_JSON_TEST = ''
SEQLABEL_ANNOTS_ES_JSON_TEST = ''

SEQLABEL_ANNOTS_EN_JSON_DEV = ''

# SPACY DATASETS
SPACY_DSET_EN = ''
SPACY_DSET_ES = ''

SPACY_DSET_ALLSPANS_EN = ''
SPACY_DSET_ALLSPANS_ES = ''



# dictionaries
LWIC_ENGLISH_DICT_PATH = ''
LWIC_SPANISH_DICT_PATH = ''

VIOLENT_WORDS_ES = ''
VIOLENT_WORDS_EN = ''

# MISC

AUTOGOLD_ANNOTS_EN_RAW = SEQLABEL_ANNOTS_EN_RAW
AUTOGOLD_ANNOTS_ES_RAW = SEQLABEL_ANNOTS_ES_RAW


OPENAI_API_KEY = ''