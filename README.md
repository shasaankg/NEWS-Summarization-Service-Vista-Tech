# News Summarization Service

An English news summarization mini-project using pretrained DistilBART,
PyTorch, Hugging Face Transformers, and FastAPI. Built in the same top-level
layout as the Fashion MNIST Classifier project.

## Completed results

| Metric | Result |
|---|---:|
| Test articles evaluated | 500 |
| ROUGE-1 F1 | **44.66** |
| ROUGE-2 F1 | **21.03** |
| ROUGE-L F1 | **30.88** |
| Mean warm GPU inference latency | **0.859 s/article** |
| Articles truncated at 1,024 tokens | 142 (28.4%) |

ROUGE is reported on a 0–100 scale. Evaluation ran on an RTX 4060 Laptop GPU.
See [the evaluation report](reports/evaluation_report.md) for confidence
intervals and [the qualitative review](reports/qualitative_review.md) for
observed omissions and factual errors.

## Approved scope

- Clean and tokenize pasted article text; truncate at the model input limit.
- Generate abstractive summaries with `sshleifer/distilbart-cnn-12-6`.
- Provide a local API and interactive Swagger demo.
- Evaluate ROUGE-1, ROUGE-2, and ROUGE-L on 500 seeded examples from the
  CNN/DailyMail 3.0.0 official test split.
- Deliver sample articles and actual generated summaries, an executed notebook,
  evaluation results, and a project report.

No training, fine-tuning, URL scraping, multilingual support, cloud deployment,
or other optional extensions are included. Git history is local.

## Project structure

```text
NEWS Summarization service/
|-- .venv/                  # Conda environment, ignored
|-- data/                   # Dataset notes, manifest, ignored downloaded data
|-- notebooks/              # Executed walkthrough
|-- src/news_summarization/ # Preprocessing, model, API, evaluation
|-- models/                 # Model manifest and ignored pretrained weights
|-- reports/                # Results, sample summaries, figures, project report
|-- tests/                  # Preprocessing/API tests and real inference check
|-- environment.yml
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Run the delivered project

Open PowerShell or Anaconda Prompt in this folder. The local environment and
downloaded assets are already prepared in the delivered copy:

```powershell
conda activate .\.venv
python -m uvicorn news_summarization.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs**, expand **POST /summarize**, select **Try it out**,
paste an English news article into `article`, and click **Execute**. A working
example is prefilled. The model loads once at server startup; wait for
`Application startup complete`. Stop the server with Ctrl+C.

If Conda is not on PATH, use Anaconda Prompt or invoke this environment directly:

```powershell
.\.venv\python.exe -m uvicorn news_summarization.api:app --host 127.0.0.1 --port 8000
```

From a second PowerShell window, test the running API:

```powershell
$article = 'The city council opened a new public library on Monday. The building includes reading rooms, computers, and a children''s section. Residents can borrow books for free. Officials said the library will open six days a week and offer classes beginning next month.'
$body = @{ article = $article; min_new_tokens = 30; max_new_tokens = 128 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/summarize' -Method Post -ContentType 'application/json' -Body $body
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
```

The response includes `summary`, `model`, `device`, `input_tokens`,
`used_input_tokens`, `truncated`, `output_tokens`, and `elapsed_seconds`.
The API accepts **20 or more words**, up to **100,000 characters**.
Generation requires `5 <= min_new_tokens < max_new_tokens <= 256`.
Token limits are not word counts. The input is truncated to 1,024 model tokens
including special tokens, and the response identifies when this happens.
Invalid requests return HTTP 422. Missing weights fail startup with the
preparation command; inference errors return HTTP 503.

The API loads local assets and performs no inference-time download. Swagger UI
loads its JavaScript/CSS from a CDN, so its browser page needs internet access;
the JSON endpoints work offline after setup.

## Recreate the environment and assets

For a fresh checkout, run these commands from the project root:

```powershell
conda env create --prefix .\.venv --file environment.yml
conda activate .\.venv
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m ipykernel install --sys-prefix --name news-summarization --display-name 'Python (News Summarization)'
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
python -m news_summarization.prepare
```

For CPU-only machines, replace `cu128` with `cpu` in the PyTorch install command.
The service automatically chooses CUDA if available, otherwise CPU. GPU uses
float16 and CPU uses float32. No paid API or Hugging Face token is required.
Allow several GB for dependencies plus approximately 1.22 GB of model weights
and 30 MB of test data. Python 3.11 is specified in `environment.yml`.

Direct dependencies are pinned in `requirements.txt`; the installed environment
snapshot is in `reports/environment.txt`. Use Conda rather than Python `venv`.
The existing Conda executable on the development machine is
`C:\Users\Shasaank\anaconda3\Scripts\conda.exe`.

## Notebook

```powershell
python -m jupyterlab
```

Open [the executed notebook](notebooks/news_summarization.ipynb), select
**Python (News Summarization)**, then restart the kernel and run all cells.
It demonstrates real model inference, truncation, API calls, sample data,
and saved evaluation results. The full 500-article evaluation is a separate
command so rerunning the walkthrough does not rerun the entire benchmark.

Headless execution:

```powershell
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/news_summarization.ipynb
```

## Evaluation and reports

```powershell
python -m news_summarization.evaluate --resume
python -m news_summarization.report
```

The evaluator uses the approved 500-article test sample (seed 42). It saves each
completed prediction to an ignored progress file. `--resume` reuses compatible
progress and rejects mismatched data, model, code, or settings. To run entirely
afresh, archive `reports/evaluation_progress.jsonl` before running evaluation.
An interrupted partial last line must be removed from that progress file before
resuming. Do not edit the final prediction records by hand.

- [Project report](reports/project_report.md): approach, architecture, results, learnings.
- [Evaluation report](reports/evaluation_report.md): protocol, scores, intervals, latency, limitations.
- [Sample articles and summaries](reports/sample_summaries.md): five complete examples.
- [Qualitative review](reports/qualitative_review.md): inspection of generated claims.
- [Metrics](reports/metrics.json) and [all predictions](reports/predictions.jsonl).
- [Verification record](reports/verification.json).

ROUGE scores are mean per-article F1 on a 0–100 scale, with stemming and 95%
bootstrap intervals. ROUGE-L is not ROUGE-Lsum. Scores on this subset are not
directly comparable with full-test upstream benchmarks. No test-based parameter
tuning or additional model comparison was performed.

## Tests

```powershell
python -m pytest -m 'not integration' -q
python -m pytest -m integration -q
```

The first suite checks cleaning, request validation, API failures, and ROUGE
calculations without loading model weights. The integration test requires the
prepared model and checks actual summaries, repeatability, and truncation.
`python -m pip check` verifies dependency consistency.

## Limitations and scope

Summaries can omit facts or introduce errors; compare them with the original
article. Truncation may discard important later details. CNN/DailyMail contains
older English articles from two publishers, and this is an in-domain evaluation.
The API runs one model instance with serialized inference and is intended as a
local course-project demo. CPU inference is slower. Exact outputs and latency
can differ across devices and dependency versions.

The large environment, downloaded dataset, and model weights are ignored by
Git. Manifests, source, notebook, results, and reports are tracked. The Git
repository has no remote configured; publishing remains with the project owner.

## References

- [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [DistilBART checkpoint](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- [Hugging Face BART documentation](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/bart)
- [PyTorch installation](https://pytorch.org/get-started/previous-versions/)
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge)
