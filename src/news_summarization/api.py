"""Start with: python -m uvicorn news_summarization.api:app --host 127.0.0.1"""

from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MAX_ARTICLE_CHARS, MODEL_ID
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "article": (
            "The city council opened a new public library on Monday. The building includes "
            "reading rooms, computers, and a children's section. Residents can borrow books "
            "for free. Officials said the library will open six days a week and offer "
            "classes beginning next month."
        ),
        "min_new_tokens": 30, "max_new_tokens": 128,
    }]})
    article: Annotated[str, Field(min_length=1, max_length=MAX_ARTICLE_CHARS)]
    min_new_tokens: Annotated[int, Field(strict=True, ge=5, le=255)] = 30
    max_new_tokens: Annotated[int, Field(strict=True, ge=6, le=256)] = 128

    @model_validator(mode="after")
    def validate_lengths(self):
        if self.min_new_tokens >= self.max_new_tokens:
            raise ValueError("min_new_tokens must be less than max_new_tokens")
        return self


class SummaryResponse(BaseModel):
    summary: str
    model: str
    device: str
    input_tokens: int
    used_input_tokens: int
    truncated: bool
    output_tokens: int
    elapsed_seconds: float


def create_app(summarizer_factory=Summarizer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup explicitly if assets are missing; never silently fake summaries.
        app.state.summarizer = summarizer_factory()
        yield
        app.state.summarizer = None

    app = FastAPI(
        title="News Summarization Service", version="1.0.0",
        description=("Summarize pasted English news articles using local DistilBART. "
                     "Inputs beyond 1,024 model tokens are truncated. "
                     "Generated summaries can omit or misstate details; check the source."),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ready", "model": MODEL_ID, "device": app.state.summarizer.device}

    @app.post("/summarize", response_model=SummaryResponse)
    def summarize(request: SummaryRequest):
        try:
            result = app.state.summarizer.summarize(
                request.article, request.min_new_tokens, request.max_new_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Summarization failed")
            raise HTTPException(status_code=503, detail="Model inference failed; see server logs.") from exc
        return result.to_dict()

    return app


app = create_app()
