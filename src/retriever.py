from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "education_docs",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name)
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(self.persist_dir))
        )
        self.collection = self._get_or_create_collection()

    def _collection_exists(self) -> bool:
        return any(c.name == self.collection_name for c in self.client.list_collections())

    def _get_or_create_collection(self) -> chromadb.api.models.Collection:
        if self._collection_exists():
            return self.client.get_collection(name=self.collection_name)
        return self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        self.persist()

    def persist(self) -> None:
        self.client.persist()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {
                "document": doc,
                "metadata": metadata,
                "distance": distance,
            }
            for doc, metadata, distance in zip(docs, metadatas, distances)
        ]

    def reset(self) -> None:
        if self._collection_exists():
            self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
