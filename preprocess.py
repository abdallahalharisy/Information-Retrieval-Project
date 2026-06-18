# preprocess.py
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure required NLTK resources are available
_NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4'
}

for name, path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(name)
        except Exception:
            pass

# Prepare common tools
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS_EN = set(stopwords.words('english'))


def _get_wordnet_pos(treebank_tag):
    if not treebank_tag:
        return wordnet.NOUN
    tag = treebank_tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text,
                    language='english',
                    lowercase=True,
                    remove_punctuation=True,
                    remove_stopwords=True,
                    do_stemming=False,
                    do_lemmatize=False,
                    do_pos_tag=False,
                    do_spell_check=False):
    """Flexible preprocessing pipeline for queries and documents."""
    if not text:
        return []

    text_proc = text.lower() if lowercase else text

    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        text_proc = text_proc.translate(translator)

    tokens = word_tokenize(text_proc)
    tokens = [t for t in tokens if t.isalpha()]

    if remove_stopwords and language.lower().startswith('en'):
        tokens = [t for t in tokens if t not in _STOPWORDS_EN]

    pos_tags = None
    if do_pos_tag or do_lemmatize:
        try:
            pos_tags = pos_tag(tokens)
        except Exception:
            pos_tags = [(t, None) for t in tokens]

    if do_lemmatize:
        lemmatized = []
        for word, tag in pos_tags:
            wn_tag = _get_wordnet_pos(tag)
            lemmatized.append(_LEMMATIZER.lemmatize(word, wn_tag))
        tokens = lemmatized

    if do_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    if do_spell_check:
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            corrected = []
            misspelled = spell.unknown(tokens)
            for t in tokens:
                if t in misspelled:
                    c = spell.correction(t)
                    corrected.append(c if c is not None else t)
                else:
                    corrected.append(t)
            tokens = corrected
        except Exception:
            pass

    if do_pos_tag:
        return {'tokens': tokens, 'pos_tags': pos_tags}

    return tokens


def clean_and_preprocess(raw_text):
    """Document pipeline aligned with query preprocessing (without synonym/spell extras)."""
    return preprocess_text(raw_text,
                           lowercase=True,
                           remove_punctuation=True,
                           remove_stopwords=True,
                           do_lemmatize=True,
                           do_stemming=True)


def apply_history_boost(query_text, history_queries, max_past_queries=5, boost_repeat=1):
    """Weight query terms using recent search history."""
    if not query_text or not history_queries:
        return query_text

    from collections import Counter
    term_counts = Counter()
    for past in history_queries[-max_past_queries:]:
        term_counts.update(past.split())

    tokens = query_text.split()
    token_set = set(tokens)
    for term, count in term_counts.most_common(8):
        if term in token_set:
            tokens.extend([term] * min(count, boost_repeat))

    return ' '.join(tokens)


def _expand_query_synonyms(tokens, max_synonyms_per_term=1):
    expanded = []
    for token in tokens:
        synonyms = []
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != token and name.isalpha():
                    synonyms.append(name)
        if synonyms:
            expanded.extend(synonyms[:max_synonyms_per_term])
    return expanded


def refine_query_text(text,
                      expand_synonyms=True,
                      do_spell_check=True,
                      max_synonyms_per_term=1,
                      history_queries=None):
    """Preprocess and refine a query using the same base pipeline as documents."""
    if not text:
        return ''

    tokens = preprocess_text(text,
                             lowercase=True,
                             remove_punctuation=True,
                             remove_stopwords=True,
                             do_lemmatize=True,
                             do_pos_tag=False,
                             do_spell_check=do_spell_check)

    if expand_synonyms:
        tokens.extend(_expand_query_synonyms(tokens, max_synonyms_per_term=max_synonyms_per_term))

    tokens = [_STEMMER.stem(t) for t in tokens]
    query_text = ' '.join(tokens)
    return apply_history_boost(query_text, history_queries)


if __name__ == '__main__':
    sample = "The boys are running and the leaves are falling."
    print('clean_and_preprocess:', clean_and_preprocess(sample))
    print('lemmatize + pos:', preprocess_text(sample, do_lemmatize=True, do_pos_tag=True))
    print('refine_query:', refine_query_text('running bird', expand_synonyms=True))
