import random

from spacy.tokens import Doc, Token
from urllib.parse import urlparse
import regex as re
from typing import Union

from data_tools.classif_data.data_loaders import corpus_telegram_en_ph1, corpus_telegram_es_ph1
from data_tools.spacy_utils import create_spacy_model
from data_tools.seqlab_data.seqlab_utils import check_tokenization_custom

URL_REGEX = r'^[\w-]+(\.[\w-]+)+'

# Regular Expression for URL
URL_PATTERN = re.compile(
    r'^(?:[a-z]+://)?'   # Protocol (optional)
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # ...or localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or IP
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$',
    re.IGNORECASE
)

URL_PATTERN_UNANCHORED = re.compile(
    r'(?:[a-z]+://)?'   # Protocol (optional)
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # ...or localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or IP
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)',
    re.IGNORECASE
)

#HASHTAG_PATTERN = r'^(#)([a-zA-Z0-9_]+)$'  # Matches hashtags
HASHTAG_PATTERN = r'^(#)([\p{L}0-9_]+)$' # hashtags, with all unicode letters

#MENTION_PATTERN = r'^(@)([a-zA-Z0-9_]+)$'  # Matches mentions according to Twitter's specifications
MENTION_PATTERN = r'^(@)([\p{L}0-9_]+)$'  # Matches mentions according to Twitter's specifications

def is_underscored_text(s):
    pattern = r'^\p{L}[\p{L}\p{N}]*(_[\p{L}\p{N}]+)+$'
    # Check if the entire string matches the pattern
    if re.match(pattern, s, re.UNICODE):
        # Split the string at underscores and also keep the underscores as tokens
        tokens = re.split(r'(_)', s)
        # Removing any empty strings that may result from the split
        tokens = [token for token in tokens if token]
        return True, tokens

    return False, []

def unicode_code_points(s:str):
    return ';'.join([f"U+{ord(char):04X}" for char in s])

def is_url(token: Union[str, Token]):
    if isinstance(token, Token):
        return token.like_url or URL_PATTERN.match(token.text) #re.search(URL_REGEX, token.text)
    else: return URL_PATTERN.match(token) # re.search(URL_REGEX, token)


def detect_hashtag_or_mention(text):
    hashtag_match = re.match(HASHTAG_PATTERN, text)
    if hashtag_match: return True, hashtag_match.group(1), hashtag_match.group(2)
    mention_match = re.match(MENTION_PATTERN, text)
    if mention_match: return True, mention_match.group(1), mention_match.group(2)
    return False, None, None

class SpacyURLTokenizer:
    '''
    Spacy tokenizer that splits urls into subtokens, based on an existing Language object,
    it adds new tokens to the vocab from the url and url-like tokens.
    It also performs an analogous tokenization mentions and hashtags.
    '''

    def __init__(self, nlp):
        self.vocab = nlp.vocab
        self.original_tokenizer = nlp.tokenizer

    def __call__(self, text):
        doc = self.original_tokenizer(text)
        tokens_extended = [] # orig. tokens with new url, mention and hashtag subtokens
        new_tokens = [] # only newly added tokens
        for token in doc:
            if is_url(token): # split url and add subtokens
                url_tokens = tokenize_url(token.text)
                tokens_extended.extend(url_tokens)
                new_tokens.extend(url_tokens)
            else:
                is_mention_or_hashtag, prefix, token_text = detect_hashtag_or_mention(token.text)
                if is_mention_or_hashtag:
                    mh_tokens = [prefix] + tokenize_url_subpart(token_text)
                    tokens_extended.extend(mh_tokens)
                    new_tokens.extend(mh_tokens)
                elif is_email(token.text):
                    email_tokens = tokenize_url_subpart(token.text)
                    tokens_extended.extend(email_tokens)
                    new_tokens.extend(email_tokens)
                elif is_underscored_text(token.text)[0]:
                    underscored_tokens = is_underscored_text(token.text)[1]
                    tokens_extended.extend(underscored_tokens)
                    new_tokens.extend(underscored_tokens)
                else: tokens_extended.append(token.text)
        # add new tokens to vocab
        for word in new_tokens:
            if word not in self.vocab.strings:
                self.vocab.strings.add(word)
        # create new doc with extended tokens
        return Doc(self.vocab, words=tokens_extended)

def tokenize_url_subpart(subpart):
    ''' Universal solution to tokenize either netloc, path, or query part of an url:
        split into sequences of alphanumerics, and non-alphanumerics,
        with '_' needing to be defined as non-alpha separately
     '''
    #return re.findall(r'[a-zA-Z0-9]+|_+|[^a-zA-Z0-9_]+', subpart)
    return re.findall(r'[\p{L}0-9]+|_+|[^\p{L}0-9_]+', subpart)

def is_email(s):
    # A basic email validation pattern (not exhaustive but should work for many common email addresses)
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, s))

def tokenize_url(url: str):
    # check if the url starts with any scheme: a sequence of letters followed by ://
    scheme_added = False
    if not re.match(r'^\w+://', url):
        url = 'https://' + url # add scheme to parse correctly with urlparse
        scheme_added = True
    try:
        url_parts = urlparse(url) # Parse the url
    except ValueError as e:
        print(f'WARNING: URL parsing error for url:\n[{url}]')
        print(e)
        return [url]
    tokens = []
    if url_parts.scheme:
        if not scheme_added: tokens.extend([url_parts.scheme, '://'])
    else:
        # it should be processed ok, this is due to deformed schema string,
        # but it should be put into path and tokenized correctly
        print(f'WARNING: URL scheme is empty: [{url}]')
        #print(f'netloc: [{url_parts.netloc}], path: [{url_parts.path}], query: [{url_parts.query}]')
    tokens.extend(tokenize_url_subpart(url_parts.netloc))
    #print('\t', ';'.join(t for t in tokenize_url_path(url_parts.path)))
    tokens.extend(tokenize_url_subpart(url_parts.path))
    # Add the query string if it exists, splitting on '&'
    if url_parts.query:
        tokens.append('?')
        tokens.extend(tokenize_url_subpart(url_parts.query))
    # Filter out any remaining empty tokens
    tokens = [token for token in tokens if token]
    return tokens

url_test_sentences = [
    "Visit our website at https://www.example.com/path/to/this-is-a-very-long-path-that-ends-with-a-sentence for more information",
    "For additional resources, check out http://www.foo.bar/path/to/here_is_another_long_path_that_ends_with_a_different_sentence",
    "To understand the problem, see the examples at https://github.com/openai/gpt-3/examples/are.very.helpful.for.understanding.the.problem",
    "What do you think about the content at http://foo.bar.baz.com/path/to/and-another-level_for_good_measure.what_do_you_think?",
    "The most popular items on the website are not always the best sellers, see https://www.amazon.com/Best-Sellers/not_always_the_most.popular.items_on_the.website",
    "This random sentence has no meaning: ftp://file.example.com/resource/this-is-just.a.random.sentence_with_no.meaning",
    "My favorite hobby is playing music. You can listen to some of my favorite songs at https://www.youtube.com/watch/playing_music_is_my_favorite.hobby",
    "Local servers are great for development. Learn more at http://localhost:8080/path/to/local.servers.are.great-for.development",
    "Reading Wikipedia articles is a great way to learn new things. Check this out: https://en.wikipedia.org/wiki/reading.wikipedia.articles-is.a.great.way_to_learn.new.things",
    "This subreddit includes posts from all over Reddit: https://www.reddit.com/r/all/this-subreddit_includes.posts.from.all.over.reddit"
]

def test_native_url_tokenizer(lang='en'):
    # load the spacy model for the language
    nlp = create_spacy_model(lang)
    for url in url_test_sentences:
        doc = nlp(url)
        print(url)
        res = nlp.tokenizer.explain(url)
        url_match = False
        for typ, tok in res:
            if typ == 'URL_MATCH':
                print(f'URL_MATCH: {tok}')
                url_match = True
        if not url_match: print('No URL_MATCH')
        print()
        #assert spacy_tok_check(doc, url)

def test_tokenize_url(lang='en'):
    # create a custom tokenizer from blank english spacy model, and use it to tokenize the test urls,
    # and check that the tokenization equals the original text
    nlp = create_spacy_model(lang) #spacy.blank(lang)
    nlp_custok = create_spacy_model(lang) #spacy.blank(lang)
    nlp_custok.tokenizer = SpacyURLTokenizer(nlp_custok)
    for url in url_test_sentences:
        doc = nlp(url)
        print('||'.join([token.text for token in doc]))
        doc_cust = nlp_custok(url)
        print('||'.join([token.text for token in doc_cust]))
        print()
        assert check_tokenization_custom(doc_cust, url)

def test_tokenizer_on_corpus(lang, sample=None, rseed=42892):
    if lang == 'en': cdf = corpus_telegram_en_ph1()
    elif lang == 'es': cdf = corpus_telegram_es_ph1()
    texts = cdf['body'].tolist()
    if sample:
        random.seed(rseed)
        texts = random.sample(texts, sample)
    # test the custom tokenizer on the corpus
    nlp = create_spacy_model(lang)
    nlp_cust = create_spacy_model(lang)
    nlp_cust.tokenizer = SpacyURLTokenizer(nlp_cust)
    N = len(texts)
    for i, txt in enumerate(texts):
        doc = nlp_cust(txt)
        assert check_tokenization_custom(doc, doc.text), f'test_fail:\n[{doc.text}]\n{";".join([token.text for token in doc])}'
        if URL_PATTERN_UNANCHORED.search(txt):
            print(f'[{txt}]')
            print('||'.join([token.text for token in nlp(txt)]))
            print('||'.join([token.text for token in doc]))
            print()
        if (i+1) % 100 == 0: print(f'\ntext processed: {i+1}/{N}\n')

if __name__ == '__main__':
    #test_tokenize_url()
    #test_native_url_tokenizer()
    #custom_tokenizer_test()
    test_tokenizer_on_corpus('es', sample=None)