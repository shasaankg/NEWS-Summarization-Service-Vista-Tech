from fastapi.testclient import TestClient
import pytest

from news_summarization.api import create_app
from news_summarization.preprocessing import clean_article
from news_summarization.summarizer import SummaryResult

ARTICLE = "The city council opened a library on Monday. " * 4


class FakeSummarizer:
    device = "cpu"

    def summarize(self, article, min_new_tokens, max_new_tokens):
        clean_article(article)
        return SummaryResult("A library opened.", "test-double", "cpu", 40, 40, False, 4, 0.01)


@pytest.fixture
def client():
    with TestClient(create_app(FakeSummarizer)) as value:
        yield value


def test_health_and_valid_request(client):
    assert client.get("/health").json()["status"] == "ready"
    response = client.post("/summarize", json={"article": ARTICLE})
    assert response.status_code == 200
    assert response.json()["summary"] == "A library opened."
    assert response.json()["truncated"] is False


@pytest.mark.parametrize("body", [
    {}, {"article": " "}, {"article": "short"}, {"article": 123},
    {"article": ARTICLE, "min_new_tokens": 128, "max_new_tokens": 30},
    {"article": ARTICLE, "max_new_tokens": 257},
    {"article": ARTICLE, "min_new_tokens": 4},
    {"article": ARTICLE, "max_new_tokens": "100"},
    {"article": ARTICLE, "min_new_tokens": True},
    {"article": ARTICLE, "unexpected": "field"},
    {"article": "a" * 100001},
])
def test_invalid_requests_return_422(client, body):
    assert client.post("/summarize", json=body).status_code == 422


def test_inference_errors_return_503_without_leaking_details():
    class Broken(FakeSummarizer):
        def summarize(self, *args):
            raise RuntimeError("private diagnostic details")
    with TestClient(create_app(Broken)) as client:
        response = client.post("/summarize", json={"article": ARTICLE})
        assert response.status_code == 503
        assert "private" not in response.text


def test_missing_model_fails_startup():
    def missing():
        raise FileNotFoundError("Model missing")
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(missing)):
            pass
