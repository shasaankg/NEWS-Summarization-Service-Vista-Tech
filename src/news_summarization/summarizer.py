"""One local model shared by the API, notebook, and evaluation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import (
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_MIN_NEW_TOKENS, GENERATION,
    MAX_INPUT_TOKENS, MODEL_DIR, MODEL_ID, SEED,
)
from .preprocessing import clean_article


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    model: str
    device: str
    input_tokens: int
    used_input_tokens: int
    truncated: bool
    output_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class Summarizer:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError("Model missing. Run: python -m news_summarization.prepare")
        torch.manual_seed(SEED)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(device).eval()
        self.input_limit = min(MAX_INPUT_TOKENS, self.model.config.max_position_embeddings)
        self._lock = Lock()

    def summarize(
        self, article: str,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> SummaryResult:
        if (
            type(min_new_tokens) is not int or type(max_new_tokens) is not int
            or not 5 <= min_new_tokens < max_new_tokens <= 256
        ):
            raise ValueError("Use 5 <= min_new_tokens < max_new_tokens <= 256.")
        cleaned = clean_article(article)
        # Serialize inference so concurrent API calls cannot multiply GPU memory use.
        with self._lock, torch.inference_mode():
            start = perf_counter()
            all_ids = self.tokenizer.encode(cleaned, add_special_tokens=True, truncation=False)
            encoded = self.tokenizer(
                cleaned, return_tensors="pt", truncation=True,
                max_length=self.input_limit,
            ).to(self.device)
            generated = self.model.generate(
                **encoded, **GENERATION,
                min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
            )
            summary = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if not summary:
                raise RuntimeError("The model returned an empty summary.")
            return SummaryResult(
                summary=summary, model=MODEL_ID, device=self.device,
                input_tokens=len(all_ids),
                used_input_tokens=encoded["input_ids"].shape[1],
                truncated=len(all_ids) > self.input_limit,
                output_tokens=len(self.tokenizer.encode(summary, add_special_tokens=False)),
                elapsed_seconds=round(perf_counter() - start, 4),
            )
