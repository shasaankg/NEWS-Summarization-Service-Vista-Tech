import pytest
from fastapi.testclient import TestClient

from news_summarization.api import create_app
from news_summarization.summarizer import Summarizer

ARTICLE = (
    "The city council opened a new public library on Monday. The building includes reading rooms, "
    "computers, and a children's section. Residents can borrow books for free. Officials said "
    "the library will open six days a week and offer classes beginning next month."
)


@pytest.mark.integration
def test_real_model_api_and_long_input():
    engine = Summarizer()
    with TestClient(create_app(lambda: engine)) as client:
        first = client.post("/summarize", json={"article": ARTICLE}).json()
        assert first["summary"] and not first["truncated"]
        repeated = client.post("/summarize", json={"article": ARTICLE}).json()
        assert repeated["summary"] == first["summary"]
        long = client.post("/summarize", json={"article": ARTICLE * 30}).json()
        assert long["truncated"] is True
        assert long["used_input_tokens"] == 1024
        assert long["input_tokens"] > 1024
        assert long["summary"]
