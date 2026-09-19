import json
import re
from pathlib import Path
from typing import Protocol

import boto3
from opensearchpy import OpenSearch

from app.core.config import Settings


class Retriever(Protocol):
    def search(self, query: str, limit: int = 4) -> list[dict]: ...


class LocalSportsRetriever:
    """Simple lexical retriever used for development and repeatable tests."""

    def __init__(self):
        path = Path(__file__).parents[1] / "data" / "sports_knowledge.json"
        self.documents = json.loads(path.read_text())

    def search(self, query: str, limit: int = 4) -> list[dict]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for document in self.documents:
            text_terms = set(re.findall(r"[a-z0-9]+", document["content"].lower()))
            score = len(terms & text_terms) / max(len(terms), 1)
            if score:
                scored.append({**document, "score": round(score, 3)})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


class OpenSearchKnnRetriever:
    """Callable tool for HNSW/cosine vector indices; it is never invoked automatically."""

    def __init__(self, settings: Settings):
        if not settings.opensearch_url:
            raise ValueError("OPENSEARCH_URL is required when RETRIEVER_BACKEND=opensearch")
        self.settings = settings
        self.client = OpenSearch(hosts=[settings.opensearch_url])
        self.bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def search(self, query: str, limit: int = 4) -> list[dict]:
        embedding = self.bedrock.invoke_model(
            modelId=self.settings.embedding_model_id,
            body=json.dumps({"inputText": query}),
        )
        vector = json.loads(embedding["body"].read())["embedding"]
        result = self.client.search(
            index=self.settings.opensearch_index,
            body={"size": limit, "query": {"knn": {self.settings.opensearch_vector_field: {"vector": vector, "k": limit}}}},
        )
        return [
            {"id": hit["_id"], "source": hit["_source"].get("source", "OpenSearch"), "content": hit["_source"]["content"], "score": hit["_score"]}
            for hit in result["hits"]["hits"]
        ]


def get_retriever(settings: Settings) -> Retriever:
    if settings.retriever_backend == "opensearch":
        return OpenSearchKnnRetriever(settings)
    return LocalSportsRetriever()
