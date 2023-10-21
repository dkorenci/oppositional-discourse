'''
Extraction of a list of violent words from the text.
'''
import re
from typing import List
import random

import pandas as pd
from langchain import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from corpus_analysis.violence_extract.llm_utils import get_cached_openai_chat_llm
from data_tools.classif_data.data_loaders import corpus_telegram_en_ph1, corpus_telegram_es_ph1
from data_tools.classif_data.data_utils import binary_dataset, LBL_CRITIC, LBL_CONSP, CLS_BINARY_IX, add_class_labels
from data_tools.spacy_utils import create_spacy_model

ENGLISH_NONE_LABEL = 'NONE'

ENGLISH_PROMPT = '''Given a text, list the words that create violent and aggressive atmosphere in the text, 
and that are not necessarily connected to physical or sexual violence. 

List only single words, not phrases.
Format the output as a comma-separated list of words.
If there are no such words, output 'NONE'.

TEXT: {text}
WORDS:'''

ENGLISH_CONTEXT = '''You are a social scientist, and you are studying political violence. '''

SPANISH_NONE_LABEL = 'NINGUNA'

SPANISH_PROMPT = '''Dado un texto, elabora una lista con las palabras que crean un tono agresivo o violento. 
No tienen por qué ser palabras conectadas con la violencia física o sexual, pueden ser términos relacionados indirectamente con la violencia.

Lista sólo palabras sueltas, no frases. 
Formatea la salida como una lista de palabras separadas por comas. 
Si no hay ese tipo de palabras, pon 'NINGUNA'. 

TEXTO: {text}
PALABRAS:'''

SPANISH_CONTEXT = '''Eres un científico social y estás estudiando la violencia política. '''

class Lemmatizer:

    def __init__(self, lang):
        self._lang = lang
        self.nlp = create_spacy_model(lang)

    # words that are for some reason not processed correctly (ex. tokenized) by spacy
    GLITCH_WORDS = {'en': set(['id']), 'es': set([])}

    def lemmatize_list(self, word_list: List[str]) -> List[str]:
        lemmatized_words = []
        for word in word_list:
            if word.lower() in self.GLITCH_WORDS[self._lang]:
                lemmatized_words.append(word)
                continue
            doc = self.nlp(word)
            assert len(doc) == 1
            for token in doc:
                lemmatized_words.append(token.lemma_)
        return lemmatized_words

    def lemmatize_text(self, text: str) -> List[str]:
        lemmatized_words = []
        doc = self.nlp(text)
        for token in doc:
            lemmatized_words.append(str.lower(token.lemma_))
        return lemmatized_words

class ChatLMMViolentWordsExtractor():

    def __init__(self, llm, prompt: str, context: str, none_label: str, exclude_phrases: bool = False):
        '''
        :param llm:
        :param none_label: label that indicates that there are no violent words in the text
        '''
        self._llm = llm
        self._prompt = PromptTemplate.from_template(prompt)
        self._context = context
        self._none_label = none_label
        self._exclude_phrases = exclude_phrases

    def _label_from_output(self, output):
        return output.strip().lower()

    def __call__(self, text) -> int:
        prompt = self._prompt.format_prompt(text=text)
        messages = [ SystemMessage(content=self._context) , HumanMessage(content=prompt.text) ]
        #messages = [ HumanMessage(content=prompt.text)]
        res = self._llm(messages)
        return self._clean_raw_output(res.content)

    def _clean_raw_output(self, rout: str):
        '''
        :param rout: raw LLM output
        :return: a cleaned list of words, lowercased, only alphabetic strings
        '''
        rout = rout.strip()
        if rout.lower() == self._none_label.lower(): return []
        def clean_word_or_phrase(w: str):
            w = w.strip()
            if re.search(r'\s', w): # whitespace in w, it is a phrase
                if self._exclude_phrases: return []
                else:
                    res = []
                    for w in w.split(): res.extend(clean_word_or_phrase(w))
                    return res
            else: # a single word
                assert re.search(r'\s', w) is None
                if not w.isalpha(): return []
                elif w.lower() == self._none_label.lower(): return [] # none label can sometimes be added as a word
                else: return [w.lower()]
        if ',' in rout:
            res = []
            for w in rout.split(','): res.extend(clean_word_or_phrase(w))
            return res
        else: return clean_word_or_phrase(rout)

class ViolentWordsCorpusExtractor():
    '''
    Extracts violent words from a corpus of texts, using a per-text word extractors.
    Words are lemmatized and aggregated.
    '''

    def __init__(self, word_extractor, lemmatizer: Lemmatizer, progress=True):
        '''
        :param word_extractor: a callable that takes a text and return a list of violent words, as a string
        '''
        self._wextr = word_extractor
        self._lemmatizer = lemmatizer
        self._progress = progress

    def __call__(self, corpus: List[str]) -> List[str]:
        all_words = set()
        for i, txt in enumerate(corpus):
            words = self._wextr(txt)
            words = [w.lower() for w in words]
            words = self._lemmatizer.lemmatize_list(words)
            all_words.update(words)
            if self._progress and (i+1) % 50 == 0: print(f'processed {i+1} texts')
        return list(all_words)

def violent_words_llm_extract(lang='en', sample_size=5, rnd_seed=1823, corpus_level=False, balanced=False, export=False):
    model = get_cached_openai_chat_llm(wrap=False)
    if lang == 'en':
        extractor = ChatLMMViolentWordsExtractor(model, ENGLISH_PROMPT, ENGLISH_CONTEXT, ENGLISH_NONE_LABEL, exclude_phrases=True)
        corpus = corpus_telegram_en_ph1()
    elif lang == 'es':
        extractor = ChatLMMViolentWordsExtractor(model, SPANISH_PROMPT, SPANISH_CONTEXT, SPANISH_NONE_LABEL, exclude_phrases=True)
        #extractor = ChatLMMViolentWordsExtractor(model, ENGLISH_PROMPT, ENGLISH_CONTEXT)
        corpus = corpus_telegram_es_ph1()
    else: raise ValueError(f'lang {lang} not supported')
    random.seed(rnd_seed)
    if not balanced: # random sample of all texts
        texts = list(corpus['body'])
        texts = random.sample(texts, sample_size)
    else: # take sample_size sample of critical and conspi texts separately, and merge
        add_class_labels(corpus)
        corpus = binary_dataset(corpus, [LBL_CRITIC, LBL_CONSP])
        txts_critic = random.sample(list(corpus[corpus[CLS_BINARY_IX] == 0]['body']), sample_size)
        txts_consp = random.sample(list(corpus[corpus[CLS_BINARY_IX] == 1]['body']), sample_size)
        texts = txts_critic + txts_consp
        assert len(txts_critic) == len(txts_consp) == sample_size
        assert len(texts) == 2*sample_size
    if not corpus_level: # for testing, extract words from each text separately, print results
        for txt in texts:
            print('TEXT:', txt)
            answer = extractor(txt)
            print('WORDS: ', ', '.join(w for w in answer))
            print()
    else:
        corpus_extractor = ViolentWordsCorpusExtractor(extractor, Lemmatizer(lang))
        words = corpus_extractor(texts)
        print(f'NUM WORDS: {len(words)}\nWORDS: {";".join(words)}')
        if export: # export as xlsx via pandas, with two columns 'word' and 'violent'
            if export == 'sort': words = sorted(words)
            elif export == 'shuffle': random.shuffle(words)
            df = pd.DataFrame({'word': words, 'violent': [-1]*len(words)})
            df.to_excel(f'violent_words_candidates_{lang}.xlsx', index=False)

def test_lemmatizer(lang):
    lemmatizer = Lemmatizer(lang)
    if lang == 'en': word_list = ['running', 'runs', 'run', 'ran', 'runners']
    elif lang == 'es':
        #word_list = ['corriendo', 'corre', 'correr', 'corrió', 'corredores']
        word_list = ['agresividad', 'rip', 'basura', 'daños', 'adverso', 'criminal', 'genocidio', 'arrestado', 'boicot']
    print(lemmatizer.lemmatize_list(word_list))

# write a function that outputs a graph - connected dots, given x and y coordinates of the dots
def plot_graph(x, y, x_label, y_label, title, save_path):
    import matplotlib.pyplot as plt
    # set x and y ranges to the same scale - [0, 2200]
    # set equal aspect ratio
    plt.xlim(0, 2200)
    plt.ylim(0, 2200)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.plot(x, y, marker='o', linestyle='--', color='r', label='Square')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_words_per_texts(lang='en'):
    if lang == 'en':
        plot_graph([20, 100, 1000], [90, 306, 2138], 'text sample size', 'num. words', '', 'en-words-for-texts.png')
    elif lang == 'es':
        plot_graph([20, 100, 1000], [74, 337, 1949], 'text sample size', 'num. words', '', 'es-words-for-texts.png')

if __name__ == '__main__':
    violent_words_llm_extract(lang='en', sample_size=500, corpus_level=True, balanced=True, export='shuffle')
    #test_lemmatizer('es')
    #plot_words_per_texts('es')

