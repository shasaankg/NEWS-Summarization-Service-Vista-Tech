"""Download only approved assets and save a reproducible evaluation manifest."""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
import numpy as np
import requests

from .config import (
    DATA_DIR, DATASET_CONFIG, DATASET_ID, EVALUATION_SIZE,
    MODEL_DIR, MODEL_ID, ROOT, SEED,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_weights(revision: str, expected_sha256: str) -> Path:
    """Stream the public checkpoint with bounded timeouts and verify its LFS hash."""
    destination = MODEL_DIR / "pytorch_model.bin"
    if destination.exists() and sha256(destination) == expected_sha256:
        return destination
    temporary = destination.with_suffix(".download")
    url = hf_hub_url(MODEL_ID, "pytorch_model.bin", revision=revision)
    with requests.get(url, stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        total = 0
        next_log = 50 * 1024**2
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                stream.write(chunk)
                total += len(chunk)
                if total >= next_log:
                    print(f"Model download: {total / 1024**2:.0f} MiB", flush=True)
                    next_log += 50 * 1024**2
    if sha256(temporary) != expected_sha256:
        raise ValueError("Downloaded model checksum differs from pinned upstream LFS metadata.")
    temporary.replace(destination)
    return destination


def prepare() -> None:
    api = HfApi()
    model_manifest_path = ROOT / "models" / "model_manifest.json"
    model_revision = (
        json.loads(model_manifest_path.read_text(encoding="utf-8"))["revision"]
        if model_manifest_path.exists() else api.model_info(MODEL_ID).sha
    )
    files = ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "pytorch_model.bin"]
    upstream = api.model_info(MODEL_ID, revision=model_revision, files_metadata=True)
    weight_metadata = next(item for item in upstream.siblings if item.rfilename == "pytorch_model.bin")
    model_hashes = {}
    for name in files:
        print(f"Preparing model file: {name}", flush=True)
        path = (
            download_weights(model_revision, weight_metadata.lfs.sha256)
            if name == "pytorch_model.bin" else
            Path(hf_hub_download(MODEL_ID, name, revision=model_revision, local_dir=MODEL_DIR))
        )
        model_hashes[name] = sha256(path)
    write_json(model_manifest_path, {
        "model_id": MODEL_ID, "revision": model_revision,
        "files_sha256": model_hashes, "training_in_this_project": False,
    })

    manifest_path = DATA_DIR / "evaluation_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    revision = existing["revision"] if existing else api.dataset_info(DATASET_ID).sha
    filename = f"{DATASET_CONFIG}/test-00000-of-00001.parquet"
    print("Preparing official CNN/DailyMail test split", flush=True)
    raw_path = Path(hf_hub_download(
        DATASET_ID, filename, repo_type="dataset", revision=revision,
        local_dir=DATA_DIR / "raw",
    ))
    dataset = load_dataset(
        "parquet", data_files={"test": str(raw_path)}, split="test",
        cache_dir=str(DATA_DIR / "raw" / "datasets_cache"),
    )
    if len(dataset) != 11490:
        raise ValueError(f"Expected 11,490 test rows, found {len(dataset)}")
    indices = np.random.default_rng(SEED).choice(len(dataset), EVALUATION_SIZE, replace=False).tolist()
    subset = dataset.select(indices)
    ids = subset["id"][:]
    if len(set(ids)) != EVALUATION_SIZE:
        raise ValueError("Duplicate IDs in evaluation sample.")
    if existing and (existing["indices"] != indices or existing["ids"] != ids):
        raise ValueError("Evaluation sample differs from the saved manifest.")
    processed_path = DATA_DIR / "processed" / "evaluation.jsonl"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", encoding="utf-8") as stream:
        for record in subset:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(manifest_path, {
        "dataset_id": DATASET_ID, "configuration": DATASET_CONFIG,
        "revision": revision, "split": "test", "population_size": len(dataset),
        "sample_size": EVALUATION_SIZE, "seed": SEED,
        "sampling": "numpy.random.default_rng(seed).choice(N, n, replace=False)",
        "numpy_version": np.__version__, "source_file": filename,
        "source_sha256": sha256(raw_path), "subset_sha256": sha256(processed_path),
        "indices": indices, "ids": ids,
    })
    print(f"Ready: {EVALUATION_SIZE} evaluation articles and local model.", flush=True)


if __name__ == "__main__":
    prepare()
