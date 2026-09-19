from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class Citation(BaseModel):
    source: str
    content: str
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    used_retrieval: bool
    low_confidence: bool
    retrieval_rounds: int
