import pytest

from news_summarization.evaluate import aggregate_scores, compute_scores


def test_rouge_reference_cases():
    exact = compute_scores("The council opened a library.", "The council opened a library.")
    assert all(score == pytest.approx(100) for score in exact.values())
    unrelated = compute_scores("cats chase mice", "rockets orbit planets")
    assert all(score == 0 for score in unrelated.values())


def test_aggregation_uses_per_article_f1_scale():
    result = aggregate_scores([
        {"scores": {"rouge1": 100, "rouge2": 80, "rougeL": 60}},
        {"scores": {"rouge1": 0, "rouge2": 20, "rougeL": 40}},
    ])
    assert all(metric["mean_f1"] == 50 for metric in result.values())
    assert result == aggregate_scores([
        {"scores": {"rouge1": 100, "rouge2": 80, "rougeL": 60}},
        {"scores": {"rouge1": 0, "rouge2": 20, "rougeL": 40}},
    ])


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError):
        aggregate_scores([])
