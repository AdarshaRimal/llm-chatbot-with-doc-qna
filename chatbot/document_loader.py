import os
import logging
import magic
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": PyPDFLoader,
    "text/plain": TextLoader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
    "text/markdown": UnstructuredMarkdownLoader,
    "text/html": UnstructuredHTMLLoader
}

class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file type is encountered"""

def detect_file_type(file_path: str) -> str:
    """Detect MIME type using python-magic"""
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)

def validate_documents(documents: List[Document]) -> None:
    """Ensure documents contain valid content"""
    if not documents:
        raise ValueError("No documents found in file")

    for doc in documents:
        if not doc.page_content.strip():
            raise ValueError("Document contains empty content")

def load_document(file_path: str) -> List[Document]:
    """Load document with enhanced error handling and metadata"""
    try:
        mime_type = detect_file_type(file_path)
        loader_class = SUPPORTED_MIME_TYPES.get(mime_type)

        if not loader_class:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {mime_type}. "
                f"Supported types: {list(SUPPORTED_MIME_TYPES.keys())}"
            )

        logger.info(f"Loading {mime_type} file: {file_path}")
        loader = loader_class(file_path)
        docs = loader.load()

        # Add metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["file_type"] = mime_type

        validate_documents(docs)
        return docs

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {str(e)}")
        raise

def split_documents(
        documents: List[Document],
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        separators: List[str] = None
) -> List[Document]:
    """Split documents with configurable parameters"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", "(?<=\\. )", " ", ""],
            length_function=len,
            keep_separator=True
        )

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["original_length"] = len(chunk.page_content)
            chunk.metadata.pop("source", None)

        logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Document splitting failed: {str(e)}")
        raise
