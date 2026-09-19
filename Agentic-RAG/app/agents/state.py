from typing import TypedDict


class AgentState(TypedDict, total=False):
    """The shared state passed between every graph node."""

    question: str
    rewritten_query: str
    retrieved_docs: list[dict]
    relevance_grade: str
    retry_count: int
    draft_answer: str
    final_answer: str
    needs_retrieval: bool
    low_confidence: bool
    verification_attempts: int
