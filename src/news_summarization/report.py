"""Build readable reports and figures from an actual completed evaluation."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import REPORTS_DIR, ROOT


def build_reports() -> None:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (REPORTS_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = json.loads((REPORTS_DIR / "sample_summaries.json").read_text(encoding="utf-8"))
    if len(predictions) != metrics["sample_size"] or len(predictions) != 500:
        raise ValueError("Reports require the complete approved 500-article evaluation.")
    if any(row["run_fingerprint"] != metrics["run_fingerprint"] for row in predictions):
        raise ValueError("Mixed evaluation runs cannot be reported together.")

    figures = REPORTS_DIR / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    scores = list(metrics["scores"].values())
    means = np.array([x["mean_f1"] for x in scores])
    intervals = np.array([x["bootstrap_95_ci"] for x in scores])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(labels, means, color=["#1d5268", "#268b9a", "#73b8bb"], width=0.55)
    ax.errorbar(labels, means, yerr=[means-intervals[:, 0], intervals[:, 1]-means], fmt="none", color="#172b36", capsize=6)
    ax.bar_label(bars, labels=[f"{x:.2f}" for x in means], padding=17)
    ax.set(ylim=(0, max(60, max(means) + 15)), ylabel="Mean per-article F1 (0-100)", title="DistilBART | CNN/DailyMail test sample (n=500)")
    ax.text(0, -0.18, "Error bars: 95% article bootstrap intervals, 2,000 resamples", transform=ax.transAxes, fontsize=9)
    fig.savefig(figures / "rouge_scores.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist([r["input_tokens"] for r in predictions], bins=25, color="#268b9a", edgecolor="white")
    axes[0].axvline(1024, color="#ad4937", linestyle="--", label="1,024-token limit")
    axes[0].set(xlabel="Article tokens before truncation", ylabel="Articles", title="Input length")
    axes[0].legend(fontsize=9)
    axes[1].hist([r["elapsed_seconds"] for r in predictions], bins=25, color="#1d5268", edgecolor="white")
    axes[1].set(xlabel="Seconds per article", ylabel="Articles", title="Warm inference latency")
    fig.savefig(figures / "input_lengths_and_latency.png", dpi=160)
    plt.close(fig)

    table = "| Metric | Mean F1 (0-100) | 95% bootstrap interval |\n|---|---:|---:|\n"
    for label, score in zip(labels, scores):
        low, high = score["bootstrap_95_ci"]
        table += f"| {label} | {score['mean_f1']:.2f} | {low:.2f}–{high:.2f} |\n"
    latency = metrics["latency_seconds"]
    truncated = metrics["truncated_articles"]
    config = metrics["run_config"]
    hardware = metrics["hardware"]
    evaluation_report = f"""# Evaluation report

Evaluation completed: {metrics['completed_at_utc']}.

## Dataset and protocol

CNN/DailyMail 3.0.0 official test split, 500 articles sampled without replacement
from 11,490 rows using NumPy default_rng(seed=42). Row indices, article IDs,
upstream revision, and file hashes are in `data/evaluation_manifest.json`.
No training, fine-tuning, or test-driven model selection was performed.
The checkpoint was selected for task fit and local inference before evaluation.

## Measured quality

{table}
![ROUGE results](figures/rouge_scores.png)

These are macro-averages of per-article F1, multiplied by 100. ROUGE uses
`rouge-score` with Porter stemming. ROUGE-L is the longest common subsequence
metric, not ROUGE-Lsum. Reference highlights are supplied unchanged. The
95% percentile intervals use 2,000 article-level bootstrap resamples (seed 42).
Intervals describe sampling variability within this dataset, not factual accuracy.
They do not cover changes in model, decoding, domain, or publisher mix.

## Inference and truncation

- Model: `{config['model']}` at revision `{config['model_revision']}`.
- Input limit: 1,024 tokens including special tokens; excess tail tokens are discarded.
- Output: 30–128 new tokens, 4 beams, length penalty 2, no repeated trigrams,
  early stopping enabled, sampling disabled.
- Execution: `{config['device']}`, `{config['dtype']}`, eager attention.
- GPU: {hardware['gpu']}; peak allocated GPU memory: {hardware['peak_gpu_allocated_mb']} MB.
- Mean latency: {latency['mean']:.3f} s; median: {latency['median']:.3f} s;
  p95: {latency['p95']:.3f} s.
- Total measured inference time: {latency['total_inference'] / 60:.2f} minutes.
- Truncated articles: **{truncated}/500 ({truncated/5:.1f}%)**.
- Mean input length: {metrics['input_tokens']['mean']:.2f} tokens;
  maximum: {metrics['input_tokens']['max']} tokens.

Latency includes preprocessing, tokenization, generation, and decoding, after
model loading and one synthetic warm-up. It excludes startup, network, and API
queue time. Requests are processed one at a time to bound GPU memory use.
It is a local measurement, not a production throughput benchmark.

![Length and latency distributions](figures/input_lengths_and_latency.png)

## Interpretation and limitations

ROUGE-1 measures unigram overlap; ROUGE-2 measures bigram overlap; ROUGE-L
captures overlap in ordered sequences. Valid paraphrases may score lower than
literal copying. ROUGE cannot establish factual consistency, so sample outputs
also need qualitative review. Truncation can remove important later details.
The report compares generated summaries against full reference highlights even
when the model sees only the article prefix; it does not shorten references.

The checkpoint was trained on the CNN/DailyMail task, so this is an in-domain
held-out benchmark, not evidence of broad generalization to current news.
The dataset represents older English news from two publishers (2007–2015) and
does not represent all regions, languages, or current reporting styles.
Only one model was tested; no unsupported model-comparison claims are made.

## Reproduction and audit trail

`predictions.jsonl` stores all 500 IDs, source row indices, reference summaries,
generated summaries, lengths, latencies, truncation flags, and individual scores.
`metrics.json` records configuration, hardware, software, revisions, and code
hashes. `sample_summaries.json` includes the first five randomly sampled input
articles, selected by sample order, not score. `environment.txt` captures installed
packages. Exact outputs or timings may differ across hardware and software.

Run `python -m news_summarization.prepare` then
`python -m news_summarization.evaluate --resume` and
`python -m news_summarization.report` from the activated project environment.
The evaluator verifies the data checksum and refuses to mix different runs.
For a fresh evaluation, archive the ignored progress JSONL before rerunning.

## Sources

- [Dataset card](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [Model card](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge)
"""
    (REPORTS_DIR / "evaluation_report.md").write_text(evaluation_report, encoding="utf-8")
    sample_md = ["# Sample articles and generated summaries\n",
                 "The first five articles in the seeded evaluation sample, without selection by score. "
                 "Source: CNN/DailyMail 3.0.0 test split. Publisher content remains attributed to CNN/DailyMail.\n"]
    for i, sample in enumerate(samples, 1):
        sample_md.append(
            f"## Example {i}\n\nDataset ID: `{sample['id']}`; source row: {sample['source_row_index']}.\n\n"
            f"### Input article\n\n{sample['article']}\n\n### Human reference\n\n{sample['reference']}\n\n"
            f"### Generated summary\n\n{sample['summary']}\n\n"
            f"Truncated: {sample['truncated']}; input tokens: {sample['input_tokens']}; "
            f"ROUGE-1/2/L: {sample['scores']['rouge1']:.2f} / {sample['scores']['rouge2']:.2f} / {sample['scores']['rougeL']:.2f}.\n"
        )
    (REPORTS_DIR / "sample_summaries.md").write_text("\n".join(sample_md), encoding="utf-8")
    project_report = f"""# News Summarization Service — Project report

## Problem and objective

News readers need concise summaries of long articles. This project implements
a local English abstractive summarization service using an existing Transformer
checkpoint, and evaluates its output against human-written reference highlights.

## Scope

The delivered workflow includes text cleaning, model tokenization, truncation,
pretrained inference, a FastAPI endpoint, an interactive API demo, and ROUGE
evaluation. Training from scratch, fine-tuning, URL scraping, multilingual
models, cloud deployment, and other optional extensions are excluded.

## Architecture

Pasted article → validation and whitespace/entity cleaning → BART tokenizer →
1,024-token prefix → pretrained DistilBART → decoded summary → JSON API response.

Reusable implementation lives in `src/news_summarization/`. The notebook calls
the same implementation as the API and evaluator, avoiding divergent behavior.
The top-level `data/`, `notebooks/`, `src/`, `models/`, and `reports/` structure
matches the previous Fashion MNIST project. A project-local Conda `.venv/`
contains Python 3.11 and the pinned Python dependencies.

## Model choice and preprocessing

DistilBART CNN has approximately 306 million parameters, with 12 encoder and
6 decoder layers. Its task-specific pretrained checkpoint fits a local inference
workflow and is smaller than BART-large. This project does not retrain it.
We use Hugging Face AutoTokenizer and AutoModelForSeq2SeqLM directly, and
pin Transformers 4.57.6. All model assets load from a local directory.

Cleaning normalizes Unicode, decodes HTML entities, removes invisible control
characters, and collapses whitespace while preserving capitalization and
punctuation. The service expects pasted article text, not HTML or a URL.
Inputs must contain at least 20 words and no more than 100,000 characters.
Tokenization follows the pretrained model's vocabulary. Overlong inputs are
truncated and explicitly marked in the response. Summary length is configurable
within documented bounds; the benchmark uses a single fixed configuration.

## Results

{table}
The evaluation used 500 seeded test articles. Mean warm inference latency was
{latency['mean']:.3f} seconds on {hardware['gpu']}; {truncated} articles were
truncated. See [the evaluation report](evaluation_report.md) for intervals,
settings, timings, distributions, and limitations. All figures and metrics were
computed locally from actual generated outputs.

## Extractive and abstractive summarization

Extractive methods select sentences already present in the article. This service
uses an abstractive encoder-decoder model that can generate new wording.
Abstraction can compress and combine facts, but may also introduce unsupported
claims. Copying overlap alone cannot measure quality, which is why the report
includes source articles, references, and generated summaries for inspection.

## Service and demonstration

`POST /summarize` accepts an article and output token limits, returning the
summary, model/device, token counts, truncation flag, and elapsed time.
`GET /health` indicates that startup loaded the model successfully. `/docs`
provides a working interactive Swagger UI. Invalid inputs return HTTP 422;
inference failures return HTTP 503 without exposing internal exception details.
Inference is serialized to control memory use. Bind the demo to 127.0.0.1.

## Reproducibility and deliverables

- Source: preprocessing, model wrapper, preparation, evaluation, report builder, API.
- Data: upstream revision, checksums, seed, indices, and all evaluation IDs.
- Model: original downloaded weights locally, revision and checksums in Git.
- Examples: five input articles, references, and actual generated summaries.
- Evaluation: 500 predictions, ROUGE scores, intervals, timings, and figures.
- Demo: executable FastAPI service and notebook demonstration.
- Documentation: setup, commands, evaluation report, project report, and sample review.
- Validation: preprocessing, API contract, metric calculations, real-model inference,
  long-input behavior, and notebook execution. See `verification.json` for outcomes.
- Version control: meaningful local milestone commits; no remote publishing.

## Learnings and limitations

Pretrained model inference makes an applied NLP service feasible without costly
training. The tokenizer's context limit is an application constraint, not just a
library setting. A reproducible test manifest is needed to make evaluation
auditable. Separating reusable source from notebook presentation makes it easier
to serve and test the same model behavior.

The model may omit information or generate factual errors. ROUGE measures overlap
and not truth. Older English news from two publishers limits generalization.
The 500-article sample is not the full benchmark, and reported scores must not be
presented as directly comparable with upstream full-test model-card results.
Half precision on GPU reduces memory use; exact cross-device equality is not
promised. The local API is a mini-project demo, not a production deployment.

## References

- [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [DistilBART checkpoint](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- [BART paper](https://aclanthology.org/2020.acl-main.703/)
- [Pointer-generator dataset paper](https://aclanthology.org/P17-1099/)
- [Hugging Face BART documentation](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/bart)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge)
"""
    (REPORTS_DIR / "project_report.md").write_text(project_report, encoding="utf-8")
    print("Wrote evaluation report, project report, sample summaries, and two figures.")


if __name__ == "__main__":
    build_reports()
