import os
import logging
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "models/embedding-001"


class VectorStoreManager:
    """Simplified vector store manager for compatibility"""

    def __init__(self, persist_dir="vector_store"):
        self.persist_dir = persist_dir
        self.embeddings = self._create_embeddings()
        os.makedirs(self.persist_dir, exist_ok=True)

    def _create_embeddings(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing")

        return GoogleGenerativeAIEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            google_api_key=api_key
        )

    def create_store(self, documents):
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        vector_store.save_local(self.persist_dir)
        return vector_store

    def load_store(self):
        return FAISS.load_local(
            folder_path=self.persist_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )