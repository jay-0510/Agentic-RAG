from functools import lru_cache

from fastapi import FastAPI

from app.agents.workflow import build_agentic_rag_graph
from app.core.config import get_settings
from app.models.schemas import AskRequest, AskResponse, Citation
from app.services.llm import get_llm
from app.services.retriever import get_retriever

app = FastAPI(title="Agentic Sports RAG", version="0.1.0")


@lru_cache
def get_graph():
    settings = get_settings()
    return build_agentic_rag_graph(settings, get_llm(settings), get_retriever(settings))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_graph().invoke({"question": request.question})
    documents = result.get("retrieved_docs", [])
    return AskResponse(
        answer=result["final_answer"],
        citations=[Citation(source=doc["source"], content=doc["content"], score=doc.get("score")) for doc in documents],
        used_retrieval=result.get("needs_retrieval", False),
        low_confidence=result.get("low_confidence", False),
        retrieval_rounds=result.get("retry_count", 0) + (1 if result.get("needs_retrieval") else 0),
    )
