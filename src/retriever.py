"""
Historical Case Retriever for AppleSupport Customer Inquiries.

Pipeline
--------
1. Load historical AppleSupport customer cases from data/processed/apple_support_cases.csv.
2. Use `sentence-transformers/all-MiniLM-L6-v2` to compute embeddings for customer_text.
3. Normalize embeddings and save the index (embeddings + metadata) in data/processed/.
4. Provide search(query, top_k) which embeds the query and computes cosine similarity
   against historical customer questions, returning the top matches and their actual
   historical AppleSupport responses.
"""

import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "apple_support_cases.csv"
)
DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "case_embeddings.npz"
)
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class HistoricalCaseRetriever:
    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        index_path: str = DEFAULT_INDEX_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.data_path = os.path.abspath(data_path)
        self.index_path = os.path.abspath(index_path)
        self.model_name = model_name
        self.model = None

        self.customer_texts = []
        self.response_texts = []
        self.embeddings = None

        self._load_or_build()

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _load_or_build(self):
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self.build_index()

    def build_index(self):
        """Builds normalized embeddings index from data/processed/apple_support_cases.csv."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Source data file not found: {self.data_path}")

        print(f"Loading cases from {self.data_path}...")
        customer_texts = []
        response_texts = []

        with open(self.data_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_text = row.get("customer_text", "").strip()
                r_text = row.get("response_text", "").strip()
                if c_text:
                    customer_texts.append(c_text)
                    response_texts.append(r_text)

        print(f"Loaded {len(customer_texts)} historical cases.")
        print(f"Computing embeddings with {self.model_name}...")
        model = self._get_model()

        # Compute embeddings for customer_text ONLY (not response_text)
        embeddings = model.encode(
            customer_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.customer_texts = customer_texts
        self.response_texts = response_texts
        self.embeddings = embeddings.astype(np.float32)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        print(f"Saving index to {self.index_path}...")
        np.savez_compressed(
            self.index_path,
            embeddings=self.embeddings,
            customer_texts=np.array(self.customer_texts, dtype=object),
            response_texts=np.array(self.response_texts, dtype=object),
        )
        print("Index successfully built and saved.")

    def _load_index(self):
        """Loads precomputed normalized embeddings and cases from disk."""
        print(f"Loading cached index from {self.index_path}...")
        data = np.load(self.index_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.customer_texts = data["customer_texts"].tolist()
        self.response_texts = data["response_texts"].tolist()
        print(f"Loaded index containing {len(self.customer_texts)} cases.")

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Given a new customer message, returns top_k semantically similar historical cases.
        """
        if self.embeddings is None or len(self.customer_texts) == 0:
            raise ValueError("Index is not built or empty.")

        model = self._get_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Dot product of normalized vectors equals cosine similarity
        similarities = np.dot(self.embeddings, query_embedding[0])

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "customer_text": self.customer_texts[idx],
                "response_text": self.response_texts[idx],
                "similarity": float(similarities[idx]),
            })

        return results


if __name__ == "__main__":
    test_query = "My iPhone battery is draining very quickly after the latest iOS update."

    retriever = HistoricalCaseRetriever()
    results = retriever.search(test_query, top_k=3)

    for i, res in enumerate(results, start=1):
        print(f"RESULT {i}")
        print(f"Similarity: {res['similarity']:.4f}")
        print(f"Customer: {res['customer_text']}")
        print(f"AppleSupport: {res['response_text']}")
        print()
