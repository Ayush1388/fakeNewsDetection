import re
import numpy as np


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+(?:\s+|$)")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouyAEIOUY]+")

SAID_PATTERN = re.compile(r"\bsaid\b", re.IGNORECASE)
FIRST_PERSON_PATTERN = re.compile(
    r"\b(i|we|my|our|me|us|mine|ours)\b",
    re.IGNORECASE,
)

NUM_LINGUISTIC_FEATURES = 22


def _count_syllables(word: str) -> int:
    """
    Cheap heuristic syllable counter (vowel-group counting).
    Good enough for a relative Flesch-ease signal; does not need
    a pronunciation dictionary.
    """

    word = word.lower()
    groups = VOWEL_GROUP_PATTERN.findall(word)
    count = len(groups)

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def extract_linguistic_features(text: str) -> np.ndarray:
    """
    Extract deterministic linguistic/statistical/stylometric features.

    These features are derived only from the supplied text.
    No external knowledge or labels are used.

    The first 14 features are the original character/token-level
    statistics. The remaining 8 are stylometric signals (sentence
    burstiness, journalistic quoting markers, an approximate
    readability score, and pronoun usage) that literature on fake
    news detection (n-gram/stylometric ensembles, e.g. EnsembleNet's
    SHAP analysis of "said", burstiness and Flesch reading ease) has
    found to be informative on top of contextual embeddings.
    """

    text = str(text)

    chars = len(text)
    tokens = TOKEN_PATTERN.findall(text)

    token_count = len(tokens)
    unique_tokens = len(set(token.lower() for token in tokens))

    words = [t for t in tokens if t.isalpha()]

    uppercase_chars = sum(1 for c in text if c.isupper())
    alphabetic_chars = sum(1 for c in text if c.isalpha())

    digits = sum(1 for c in text if c.isdigit())

    punctuation = sum(
        1 for c in text
        if not c.isalnum() and not c.isspace()
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    url_count = len(URL_PATTERN.findall(text))
    hashtag_count = len(re.findall(r"#\w+", text))
    mention_count = len(re.findall(r"@\w+", text))

    newline_count = text.count("\n")

    avg_word_length = (
        np.mean([len(w) for w in words])
        if words
        else 0.0
    )

    unique_ratio = (
        unique_tokens / token_count
        if token_count > 0
        else 0.0
    )

    uppercase_ratio = (
        uppercase_chars / alphabetic_chars
        if alphabetic_chars > 0
        else 0.0
    )

    digit_ratio = (
        digits / chars
        if chars > 0
        else 0.0
    )

    punctuation_ratio = (
        punctuation / chars
        if chars > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # Stylometric features.
    # ---------------------------------------------------------------

    sentences = [
        s for s in SENTENCE_SPLIT_PATTERN.split(text)
        if s.strip()
    ]

    sentence_count = max(len(sentences), 1)

    sentence_lengths = [
        len(TOKEN_PATTERN.findall(s)) for s in sentences
    ] or [token_count]

    avg_sentence_length = float(np.mean(sentence_lengths))

    # "Burstiness": how erratic sentence lengths are. Highly
    # variable, emotionally charged writing is a known fake-news
    # stylometric marker.
    sentence_length_std = (
        float(np.std(sentence_lengths))
        if len(sentence_lengths) > 1
        else 0.0
    )

    # Journalistic attribution marker ("X said ..."). Frequent use
    # correlates with authentic reporting that quotes sources.
    said_count = len(SAID_PATTERN.findall(text))

    quote_count = text.count('"') + text.count("“") + text.count("”")

    # Approximate Flesch reading-ease score. Fake news tends to be
    # written at a simpler reading level to reach a wider audience.
    total_syllables = sum(_count_syllables(w) for w in words) if words else 0

    if words and sentence_count > 0:
        flesch_reading_ease = (
            206.835
            - 1.015 * (len(words) / sentence_count)
            - 84.6 * (total_syllables / len(words))
        )
        # Clip to a sane range; degenerate short texts can blow up.
        flesch_reading_ease = float(
            np.clip(flesch_reading_ease, -100.0, 121.22)
        )
    else:
        flesch_reading_ease = 0.0

    all_caps_words = sum(
        1 for w in words if len(w) > 1 and w.isupper()
    )

    all_caps_ratio = (
        all_caps_words / len(words)
        if words
        else 0.0
    )

    first_person_count = len(
        FIRST_PERSON_PATTERN.findall(text)
    )

    first_person_ratio = (
        first_person_count / token_count
        if token_count > 0
        else 0.0
    )

    features = np.array(
        [
            chars,
            token_count,
            unique_tokens,
            unique_ratio,
            avg_word_length,
            uppercase_ratio,
            digit_ratio,
            punctuation_ratio,
            exclamation_count,
            question_count,
            url_count,
            hashtag_count,
            mention_count,
            newline_count,
            sentence_count,
            avg_sentence_length,
            sentence_length_std,
            said_count,
            quote_count,
            flesch_reading_ease,
            all_caps_ratio,
            first_person_ratio,
        ],
        dtype=np.float32,
    )

    assert features.shape[0] == NUM_LINGUISTIC_FEATURES

    return features
