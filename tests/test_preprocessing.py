import pytest

from news_summarization.config import MAX_ARTICLE_CHARS
from news_summarization.preprocessing import clean_article

ARTICLE = (
    "The city council opened a public library on Monday. Residents can borrow books "
    "and use computers free of charge during regular opening hours."
)


def test_preserves_meaning_and_normalizes_spacing():
    text = ARTICLE.replace("council", "Council\t&amp;\nMayor\u200b")
    cleaned = clean_article(text)
    assert "Council & Mayor" in cleaned
    assert "Monday." in cleaned
    assert "\n" not in cleaned


@pytest.mark.parametrize("text", ["", "\t\n", "short text", "123 " * 30, None])
def test_rejects_unusable_input(text):
    with pytest.raises(ValueError):
        clean_article(text)


def test_rejects_oversized_input():
    with pytest.raises(ValueError, match="exceeds"):
        clean_article("a" * (MAX_ARTICLE_CHARS + 1))
