import hashlib
import regex as re
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from corpus_analysis.violence_extract.llm_setup import langchain_sqlite_cache
from settings import OPENAI_API_KEY

import langchain as lc

def get_cached_openai_llm(model_name='gpt-3.5-turbo-0301'):
    lc.llm_cache = langchain_sqlite_cache()
    if 'gpt-3.5' in model_name:
        llm = OpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    else:
        llm = OpenAI(model_name=model_name,  n=1, best_of=1, openai_api_key=OPENAI_API_KEY)
    return llm

def get_cached_openai_chat_llm(model_name='gpt-3.5-turbo-0301', wrap=True):
    lc.llm_cache = langchain_sqlite_cache()
    #llm = OpenAIChat(model_name=model_name, openai_api_key=OPENAI_API_KEY, temperature=0)
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, temperature=0)
    if wrap: return LCChatWrapper(llm)
    else: return llm

def str_hash(s: str) -> str:
    '''
    Creat a good PERMANENT hash of a string, with a minuscule probability of clashes.
    Permanency means this is a pure function, depending only on the input string.
    '''
    h = hashlib.sha256(s.encode()).hexdigest()
    return f'{h}'

def extract_first_word(s: str) -> str:
    ''' If string is not a pure alphanumeric sequence, extract the first word from a string -
    everything before a whitespace or an interpunction mark. '''
    if re.match(r'^[a-zA-Z0-9]+$', s): return s
    return re.match(r'^[a-zA-Z0-9]+', s).group(0)

def label_normalize(out: str) -> str:
    '''
    Turns the output of an LLM, as well as class labels to a canonic form so that they can be matched.
    Lowercse, remove whitespace and non-alphanumerics from both borders.
    :return:
    '''
    out = out.strip().lower()
    remalnum = lambda s: re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', s)
    out = remalnum(out)
    return out

from langchain.schema import HumanMessage, AIMessage

class LCChatWrapper():
    '''
    Wrapper of langchain BaseChatModel objects that creates a simple interface with a __call__ method,
    that makes the chat model appear as a standard text-generating LLM.
    '''

    def __init__(self, chat_llm: BaseChatModel):
        self._chat_llm = chat_llm

    def __call__(self, input: str) -> str:
        messages = [ HumanMessage(content=input) ]
        res = self._chat_llm(messages)
        assert isinstance(res, AIMessage)
        return res.content