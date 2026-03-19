"""
exp_016/rag.py — Simple RAG with sentence-transformers + FAISS
"""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    def __init__(self, knowledge_dir: str | Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.facts: list[str] = []

        knowledge_dir = Path(knowledge_dir)
        for txt_file in sorted(knowledge_dir.glob("*.txt")):
            for line in txt_file.read_text().splitlines():
                line = line.strip()
                if line:
                    self.facts.append(line)

        embeddings = self.model.encode(self.facts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 3, threshold: float = 0.4) -> list[str]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        return [self.facts[i] for i, s in zip(indices[0], scores[0]) if i < len(self.facts) and s >= threshold]
