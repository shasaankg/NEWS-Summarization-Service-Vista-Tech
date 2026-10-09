# News Summarization Service

An English news summarization mini-project using pretrained DistilBART,
PyTorch, Hugging Face Transformers, and FastAPI. Built in the same top-level
layout as the Fashion MNIST Classifier project.

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

Setup, execution instructions, and measured results will be finalized after
the environment and end-to-end evaluation are verified.
