import re
import numpy as np


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def extract_linguistic_features(text: str) -> np.ndarray:
    """
    Extract deterministic linguistic/statistical features.

    These features are derived only from the supplied text.
    No external knowledge or labels are used.
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
        ],
        dtype=np.float32,
    )

    return features