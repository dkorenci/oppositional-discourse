import spacy
from spacy import displacy
from spacy.tokens import Span

AGENTS = 'A'
FACIL = 'F'
EFFECTOS = 'E'
VICTIMS = 'V'
PARTID = 'C'
OBJEC = 'O'

colors1 = {
        AGENTS: 'red',
        FACIL: 'green',
        EFFECTOS: 'grey',
        VICTIMS: 'yellow',
        PARTID: 'orange',
        OBJEC: 'blue'
}

colors2 = {
        AGENTS: '#ff704d',
        FACIL: '#ffbb33',
        EFFECTOS: '#b3b3b3',
        VICTIMS: '#80ccff',
        PARTID: '#66cc66',
        OBJEC: '#ffff80'
}

DEFAULT_LABEL_COLORS = {
    "ORG": "#7aecec",
    "PRODUCT": "#bfeeb7",
    "GPE": "#feca74",
    "LOC": "#ff9561",
    "PERSON": "#aa9cfc",
    "NORP": "#c887fb",
    "FAC": "#9cc9cc",
    "EVENT": "#ffeb80",
    "LAW": "#ff8197",
    "LANGUAGE": "#ff8197",
    "WORK_OF_ART": "#f0d0ff",
    "DATE": "#bfe1d9",
    "TIME": "#bfe1d9",
    "MONEY": "#e4e7d2",
    "QUANTITY": "#e4e7d2",
    "ORDINAL": "#e4e7d2",
    "CARDINAL": "#e4e7d2",
    "PERCENT": "#e4e7d2",
}

text1 = 'Private owned WHO with investors like Bill Gates can declare a new pandemic out of ' \
       'thin air anytime they want and the world governments ruled by their puppets as well as their media ' \
       'starts with the constant fear mongering, getting people to get their pharma companies injections ' \
       'and drugs that are magically ready in light speed, clear induction that they have been ready for ' \
       'the orchestrated fake pandemics, long before they start with the constant fear mongering by ' \
       'the media and governments. To those awake already, we know their games and agenda, ' \
       'but sadly most people fall for it, again and again and pay a hefty price, often with their health, ' \
       'lives, the loss of their loved ones. These are very evil beings, intent on destroying us regular people.'

spans1 = [('Private owned WHO', AGENTS), ('investors like Bill Gates', AGENTS),
          ('the world governments ruled by their puppets', FACIL), ('their media', FACIL),
          ('the constant fear mongering', EFFECTOS), ('people', VICTIMS),
          ('pharma companies', AGENTS), ('the constant fear mongering', EFFECTOS),
          ('the media', FACIL), ('governments', FACIL), ('those awake already', PARTID),
          ('agenda', OBJEC), ('most people', VICTIMS),
          ('pay a hefty price, often with their health, lives, the loss of their loved ones', EFFECTOS),
          ('their loved ones', VICTIMS), ('very evil beings', AGENTS), ('destroying us regular people', OBJEC),
          ('us regular people', VICTIMS)]

spans2 = [('Private owned WHO', AGENTS), ('investors like Bill Gates', AGENTS),
          ('the world governments ruled by their puppets', FACIL), ('their media', FACIL),
          ('the constant fear mongering', EFFECTOS), ('people', VICTIMS),
          ('pharma companies', AGENTS), ('the constant fear mongering', EFFECTOS),
          ('the media', FACIL), ('governments', FACIL), ('those awake already', PARTID),
          ('agenda', OBJEC), ('most people', VICTIMS),
          ('pay a hefty price, often with their health, lives, the loss of their loved ones', EFFECTOS),
          ('very evil beings', AGENTS), ('destroying us', OBJEC), ('regular people', VICTIMS)]

# Raw text string
text2 = "Private owned WHO with investors like Bill Gates can declare a new pandemic out of thin air anytime they want and the world governments ruled by their puppets as well as their media starts with the constant fear mongering, getting people to get their pharma companies injections and drugs that are magically ready in light speed, clear induction that they have been ready for the orchestrated fake pandemics, long before they start with the constant fear mongering by the media and governments. To those awake already, we know their games and agenda, but sadly most people fall for it, again and again and pay a hefty price, often with their health, lives, the loss of their loved ones. These are very evil beings, intent on destroying us regular people."

# List of tuples (span, label)
spans3 = [
    ("Private owned WHO", AGENTS),
    ("investors like Bill Gates", AGENTS),
    ("the world governments ruled by their puppets", FACIL),
    ("their media", FACIL),
    ("the constant fear mongering", EFFECTOS),
    ("people", VICTIMS),
    ("pharma companies", AGENTS),
    ("the constant fear mongering", EFFECTOS),
    ("the media", FACIL),
    ("governments", FACIL),
    ("those awake already", PARTID),
    ("agenda", OBJEC),
    ("most people", VICTIMS),
    ("pay a hefty price, often with their health, lives, the loss of their loved ones", EFFECTOS),
    ("very evil beings", AGENTS),
    ("destroying us", OBJEC),
    ("regular people", VICTIMS)
]



def get_token_span(doc, substr, start_at=None, soft=False):
    alpha = substr.isalpha()
    for ti, toki in enumerate(doc):
        tt = toki.text
        #if len(tt) < len(substr): continue
        if substr[0] != tt[0]: continue
        #if substr[0] != substr[0]: continue
        if start_at is None or ti >= start_at:
            for tj, _ in enumerate(doc):
                if ti <= tj:
                    txt = doc[ti: tj].text
                    if substr == txt:
                        #print(ti, tj)
                        #print(txt)
                        return (ti, tj)
                    if soft: # regard last-letter missed labeling mistakes as matches
                        if alpha and len(txt) > 1:
                            if substr == txt[:-1]: #or substr == txt[1:]:
                                #print(f'!MATCH: [{substr}] [{txt}]')
                                return (ti, tj)
    return None

def create_example(text: str, spans, method='ents', title=None):
    nlp = spacy.blank('en')
    doc = nlp(text)
    spacy_spans = []
    start_at = 0; start_tok = -1
    for txt, label in spans:
        #print(start_at, txt, label)
        res = get_token_span(doc, txt, start_at)
        if res != None:
            start_tok, end_tok = res
            spacy_spans.append(Span(doc, start_tok, end_tok, label))
        start_at = start_tok + 1
    if method == 'ents':
        doc.ents = spacy_spans
    elif method == 'spans':
        doc.spans["sc"] = spacy_spans
    if title is not None:
        doc.user_data["title"] = title
    return doc

def visualize_example(text: str, spans, method='ents', port=8080):
    doc = create_example(text, spans, method)
    style = 'ent' if method == 'ents' else 'span'
    displacy.serve(doc, style=style, port=port, options={'colors': colors2})

if __name__ == '__main__':
    visualize_example(text2, spans3, 'ents', 8081)