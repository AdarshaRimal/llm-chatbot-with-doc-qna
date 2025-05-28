import os
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

CUSTOM_PROMPT = PromptTemplate.from_template(
    """Use the following context to answer the question. 
If the context doesn't contain the answer, say "I couldn't find relevant information in the document."
Context: {context}
Question: {question}
Helpful Answer:"""
)


def _handle_gemini_error(e: Exception) -> str:
    """Process Gemini-specific errors"""
    if "SAFETY" in str(e):
        return "My response was blocked due to safety concerns. Please rephrase your question."
    return "Sorry, I encountered an error. Please try again."

def build_qa_chain(retriever):
    """Creates RAG chain with proper document context"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": CUSTOM_PROMPT  # Use the updated prompt
        }
    )
def get_answer(qa_chain: RetrievalQA, query: str) -> Dict[str, Any]:
    """Safe query execution with error boundaries"""
    try:
        result = qa_chain.invoke({"query": query})
        print("DEBUG - Full invoke result:", result)
        if not result or "result" not in result:
            return {"answer": "No answer generated", "sources": []}

        # Process source documents
        sources = [doc.metadata.get("source", "") for doc in result.get("source_documents", [])]
        return {
            "answer": result["result"],
            "sources": list(set(sources)),
            "error": None
        }

    except Exception as e:
        logger.error(f"Query failed: {query} - {str(e)}")
        return {
            "answer": _handle_gemini_error(e),
            "sources": [],
            "error": str(e)
        }