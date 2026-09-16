import re

from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState
from app.core.config import Settings
from app.services.llm import LLM, parse_json_response
from app.services.retriever import Retriever

GREETING_PATTERN = re.compile(r"^\s*(hi|hello|hey|thanks|thank you)[!.?\s]*$", re.I)


def _context(documents: list[dict]) -> str:
    return "\n\n".join(f"[{doc['source']}] {doc['content']}" for doc in documents)


def build_agentic_rag_graph(settings: Settings, llm: LLM, retriever: Retriever):
    """Build the agentic graph. Tool calls and loops are explicit graph decisions."""

    def router(state: AgentState) -> dict:
        question = state["question"]
        if settings.llm_backend == "local":
            needs_retrieval = not bool(GREETING_PATTERN.match(question))
        else:
            decision = parse_json_response(
                llm.complete(
                    "Decide whether this question needs an external sports knowledge base. "
                    "Return JSON only: {\"needs_retrieval\": true|false}.\n"
                    f"QUESTION: {question}"
                ),
                {"needs_retrieval": True},
            )
            needs_retrieval = bool(decision["needs_retrieval"])
        return {"needs_retrieval": needs_retrieval, "retry_count": 0, "low_confidence": False}

    def rewrite_query(state: AgentState) -> dict:
        question = state["question"]
        if settings.llm_backend == "local":
            rewritten = question
        else:
            rewritten = llm.complete(
                "Rewrite this as a concise semantic-search query. Preserve important sports terms. "
                "Return only the query.\nQUESTION: " + question
            ).strip()
        return {"rewritten_query": rewritten}

    def retrieve(state: AgentState) -> dict:
        # The retriever is a callable tool selected by the router, not a fixed pipeline stage.
        return {"retrieved_docs": retriever.search(state["rewritten_query"])}

    def grade_relevance(state: AgentState) -> dict:
        documents = state["retrieved_docs"]
        if settings.llm_backend == "local":
            grade = "pass" if documents and documents[0].get("score", 0) >= 0.15 else "fail"
        else:
            result = parse_json_response(
                llm.complete(
                    "Are these sources relevant enough to answer the question? Return JSON only: "
                    "{\"grade\": \"pass\"|\"fail\"}.\n"
                    f"QUESTION: {state['question']}\nCONTEXT:\n{_context(documents)}"
                ),
                {"grade": "fail"},
            )
            grade = result["grade"] if result.get("grade") in {"pass", "fail"} else "fail"
        return {"relevance_grade": grade}

    def reformulate(state: AgentState) -> dict:
        retries = state["retry_count"] + 1
        if settings.llm_backend == "local":
            query = f"{state['question']} rules definition"
        else:
            query = llm.complete(
                "Reformulate this failed search query with alternate terminology. Return only query.\n"
                f"QUESTION: {state['question']}\nFAILED QUERY: {state['rewritten_query']}"
            ).strip()
        return {"retry_count": retries, "rewritten_query": query}

    def generate(state: AgentState) -> dict:
        documents = state.get("retrieved_docs", [])
        if not documents:
            answer = "I could not find relevant information in the connected knowledge base."
        else:
            answer = llm.complete(
                "Answer strictly from CONTEXT. If it does not support a claim, say so. Be concise.\n"
                f"CONTEXT:\n{_context(documents)}\nQUESTION: {state['question']}"
            )
        return {"draft_answer": answer}

    def verify(state: AgentState) -> dict:
        documents = state.get("retrieved_docs", [])
        if settings.llm_backend == "local":
            supported = bool(documents)
        else:
            result = parse_json_response(
                llm.complete(
                    "Check whether ANSWER is fully supported by CONTEXT. Return JSON only: "
                    "{\"supported\": true|false}.\n"
                    f"CONTEXT:\n{_context(documents)}\nANSWER: {state['draft_answer']}"
                ),
                {"supported": False},
            )
            supported = bool(result["supported"])
        attempts = state.get("verification_attempts", 0)
        return {"verification_attempts": attempts + 1, "final_answer": state["draft_answer"], "low_confidence": not supported}

    def direct_response(_: AgentState) -> dict:
        return {"final_answer": "Hello. I can answer questions using the connected sports knowledge base.", "low_confidence": False}

    def after_router(state: AgentState) -> str:
        return "rewrite_query" if state["needs_retrieval"] else "direct_response"

    def after_grade(state: AgentState) -> str:
        if state["relevance_grade"] == "pass":
            return "generate"
        if state["retry_count"] < settings.max_retrieval_retries:
            return "reformulate"
        return "generate"

    def after_verify(state: AgentState) -> str:
        if state["low_confidence"] and state["verification_attempts"] <= settings.max_verification_retries:
            return "generate"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("reformulate", reformulate)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("direct_response", direct_response)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", after_grade)
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify)
    graph.add_edge("direct_response", END)
    return graph.compile()
