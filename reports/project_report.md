# News Summarization Service — Project report

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

| Metric | Mean F1 (0-100) | 95% bootstrap interval |
|---|---:|---:|
| ROUGE-1 | 44.66 | 43.66–45.67 |
| ROUGE-2 | 21.03 | 20.01–22.13 |
| ROUGE-L | 30.88 | 29.91–31.97 |

The evaluation used 500 seeded test articles. Mean warm inference latency was
0.859 seconds on NVIDIA GeForce RTX 4060 Laptop GPU; 142 articles were
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
