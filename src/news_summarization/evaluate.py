"""Evaluate a fixed test sample; report macro-average ROUGE F1 on a 0-100 scale."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
from rouge_score import rouge_scorer
import torch
import transformers

from .config import (
    DATA_DIR, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MIN_NEW_TOKENS, GENERATION,
    MAX_INPUT_TOKENS, MODEL_ID, REPORTS_DIR, ROOT, SEED,
)
from .prepare import sha256, write_json
from .summarizer import Summarizer

METRICS = ("rouge1", "rouge2", "rougeL")


def compute_scores(reference: str, prediction: str) -> dict:
    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    return {name: score.fmeasure * 100 for name, score in scorer.score(reference, prediction).items()}


def aggregate_scores(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation results.")
    values = np.array([[row["scores"][key] for key in METRICS] for row in rows])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(rows), size=(2000, len(rows)))
    means = values[draws].mean(axis=1)
    return {
        key: {
            "mean_f1": round(float(values[:, i].mean()), 4),
            "bootstrap_95_ci": [round(float(x), 4) for x in np.percentile(means[:, i], [2.5, 97.5])],
        }
        for i, key in enumerate(METRICS)
    }


def evaluate(resume: bool = False, device: str = "auto") -> None:
    subset_path = DATA_DIR / "processed" / "evaluation.jsonl"
    manifest = json.loads((DATA_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if sha256(subset_path) != manifest["subset_sha256"]:
        raise ValueError("Evaluation data checksum differs from preparation manifest.")
    records = [json.loads(line) for line in subset_path.read_text(encoding="utf-8").splitlines()]
    if [x["id"] for x in records] != manifest["ids"] or len(records) != 500:
        raise ValueError("Expected the approved 500-article sample in saved order.")
    model_manifest = json.loads((ROOT / "models" / "model_manifest.json").read_text(encoding="utf-8"))
    REPORTS_DIR.mkdir(exist_ok=True)
    engine = Summarizer(device=device)
    run_config = {
        "model": MODEL_ID, "model_revision": model_manifest["revision"],
        "dataset_revision": manifest["revision"], "subset_sha256": manifest["subset_sha256"],
        "generation": {**GENERATION, "min_new_tokens": DEFAULT_MIN_NEW_TOKENS,
                       "max_new_tokens": DEFAULT_MAX_NEW_TOKENS},
        "input_token_limit": MAX_INPUT_TOKENS, "device": engine.device,
        "dtype": str(engine.dtype), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "implementation_sha256": {
            name: sha256(ROOT / "src" / "news_summarization" / name)
            for name in ["config.py", "preprocessing.py", "summarizer.py", "evaluate.py"]
        },
        "rouge": {"use_stemmer": True, "aggregation": "mean per-article F1", "scale": "0-100"},
    }
    fingerprint = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    progress_path = REPORTS_DIR / "evaluation_progress.jsonl"
    rows = []
    if resume and progress_path.exists():
        rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
        if any(r["run_fingerprint"] != fingerprint for r in rows):
            raise ValueError("Cannot resume an evaluation with different data, code, model, or settings.")
        if [r["id"] for r in rows] != manifest["ids"][:len(rows)]:
            raise ValueError("Progress records do not match the approved sample order.")
    elif progress_path.exists():
        raise FileExistsError("Progress already exists. Use --resume or archive the old progress file.")

    warmup = (
        "The city council opened a new public library on Monday. The building has reading rooms, "
        "computers, and a children's section. Residents can borrow books for free. Officials said "
        "the library will open six days a week and offer classes beginning next month."
    )
    engine.summarize(warmup)
    start = perf_counter()
    with progress_path.open("a", encoding="utf-8") as stream:
        for index in range(len(rows), len(records)):
            record = records[index]
            result = engine.summarize(record["article"])
            row = {
                "sample_index": index, "source_row_index": manifest["indices"][index],
                "id": record["id"], "reference": record["highlights"],
                **result.to_dict(), "scores": compute_scores(record["highlights"], result.summary),
                "run_fingerprint": fingerprint,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if (index + 1) % 10 == 0 or index == 0:
                print(f"Evaluated {index + 1}/{len(records)} articles; latest latency {result.elapsed_seconds:.2f}s", flush=True)

    latencies = np.array([r["elapsed_seconds"] for r in rows])
    metrics = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows), "population_size": manifest["population_size"], "seed": SEED,
        "run_config": run_config, "run_fingerprint": fingerprint,
        "scores": aggregate_scores(rows),
        "bootstrap": {"resamples": 2000, "seed": SEED, "unit": "article"},
        "latency_seconds": {
            "mean": round(float(latencies.mean()), 4), "median": round(float(np.median(latencies)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "total_inference": round(float(latencies.sum()), 4),
            "latest_run_wall_time_excluding_load_and_warmup": round(perf_counter() - start, 4),
        },
        "truncated_articles": sum(r["truncated"] for r in rows),
        "input_tokens": {"mean": round(float(np.mean([r["input_tokens"] for r in rows])), 2),
                         "max": max(r["input_tokens"] for r in rows)},
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "gpu": torch.cuda.get_device_name(0) if engine.device == "cuda" else None,
                     "cuda_runtime": torch.version.cuda,
                     "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                     if engine.device == "cuda" else None},
    }
    write_json(REPORTS_DIR / "metrics.json", metrics)
    (REPORTS_DIR / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8",
    )
    # First five in the random sample, never selected by score.
    write_json(REPORTS_DIR / "sample_summaries.json", [
        {"article": records[i]["article"], **rows[i]} for i in range(5)
    ])
    print(json.dumps(metrics["scores"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    evaluate(resume=args.resume, device=args.device)
