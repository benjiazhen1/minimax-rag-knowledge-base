# MiniMax RAG Knowledge Base
# 提供 RAG 检索增强生成的完整实现

from .llm import MiniMaxLLM
from .retriever import TFIDFRetriever, FAISSRetriever
from .rag_chain import RAGChain, create_rag_chain

__all__ = [
    "MiniMaxLLM",
    "TFIDFRetriever", 
    "FAISSRetriever",
    "RAGChain",
    "create_rag_chain",
]

__version__ = "1.0.0"
