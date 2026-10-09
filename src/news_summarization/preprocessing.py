"""Conservative cleaning that preserves capitalization and punctuation."""

import html
import re
import unicodedata

from .config import MAX_ARTICLE_CHARS


def clean_article(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Article must be a string.")
    if len(text) > MAX_ARTICLE_CHARS:
        raise ValueError(f"Article exceeds {MAX_ARTICLE_CHARS:,} characters.")
    text = unicodedata.normalize("NFC", html.unescape(text))
    # Whitespace controls become spaces; remove other invisible controls.
    text = "".join(
        " " if c.isspace() else c
        for c in text
        if c.isspace() or unicodedata.category(c) not in {"Cc", "Cf"}
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not any(c.isalpha() for c in text):
        raise ValueError("Article must contain readable text.")
    if len(text.split()) < 20:
        raise ValueError("Enter an article containing at least 20 words.")
    return text
