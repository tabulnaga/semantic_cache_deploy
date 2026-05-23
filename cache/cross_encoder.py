import math
import os
from typing import Callable, Dict, List, Optional


class CrossEncoder:
    """API-based replacement for local torch CrossEncoder. Same interface."""

    def __init__(self, model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # model_name_or_path accepted for interface compat; we use OpenAI embeddings
        self.model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def pair_distance(self, query: str, context: str) -> float:
        return 1 - self.predict([query], [context])[0]

    def predict(self, queries: List[str], contexts: List[str]) -> List[float]:
        """
        Direct cross encoder prediction for query-context pairs.

        Args:
            queries: List of query strings
            contexts: List of context strings (same length as queries)

        Returns:
            List of similarity scores [0.0-1.0] for each query-context pair
        """
        client = self._get_client()

        # Deduplicate texts to minimize API calls
        all_texts = list(set(queries + contexts))
        resp = client.embeddings.create(model=self.model_name, input=all_texts)

        # Build embedding lookup
        emb_map = {}
        for i, item in enumerate(resp.data):
            emb_map[all_texts[i]] = item.embedding

        # Compute cosine similarity for each pair
        scores = []
        for q, c in zip(queries, contexts):
            q_emb = emb_map[q]
            c_emb = emb_map[c]
            sim = _cosine_sim(q_emb, c_emb)
            scores.append(sim)

        return scores

    def create_reranker(self):
        return CrossEncoderReranker(self)


class CrossEncoderReranker:
    def __init__(self, cross_encoder: CrossEncoder):
        self.cross_encoder = cross_encoder

    def __call__(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Cross encoder reranker function for semantic cache integration.

        Args:
            query: The search query
            candidates: List of cache candidate dictionaries

        Returns:
            Filtered and reordered candidates with cross encoder metadata
        """
        if not candidates:
            return []

        # Extract prompts for cross encoder scoring
        prompts = [c.get("prompt", "") for c in candidates]

        # Get cross encoder scores
        scores = self.cross_encoder.predict([query] * len(prompts), prompts)

        # Create scored candidates with metadata using dict comprehension
        validated_candidates = [
            (
                {
                    **candidate,
                    "reranker_type": "cross_encoder",
                    "reranker_score": float(score),
                    "reranker_distance": 1 - float(score),
                },
                score,
            )
            for candidate, score in zip(candidates, scores)
        ]
        # Sort by cross encoder score (highest first)
        validated_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return just the enriched candidates
        return [candidate for candidate, _ in validated_candidates]


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
