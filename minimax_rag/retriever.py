#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索器模块
支持 TF-IDF 和 FAISS 两种向量检索方式
"""

import numpy as np
from typing import List, Tuple, Optional


class TFIDFRetriever:
    """TF-IDF 检索器（轻量级，适合小数据）"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.vectors = None
        
        # 简单的 TF-IDF 实现
        self.vocab: dict = {}
        self.idf: dict = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return text.lower().split()
    
    def _compute_tf(self, tokens: List[str]) -> dict:
        """计算 TF"""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens) or 1
        for token in tf:
            tf[token] /= total
        return tf
    
    def _compute_idf(self):
        """计算 IDF"""
        N = len(self.documents)
        df = {}
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        
        for token, df_count in df.items():
            self.idf[token] = np.log(N / (df_count + 1)) + 1
    
    def add_documents(self, documents: List[str]):
        """添加文档到索引"""
        self.documents = documents
        self._compute_idf()
        
        # 构建词表
        all_tokens = set()
        for doc in documents:
            all_tokens.update(self._tokenize(doc))
        
        self.vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        
        # 构建 TF-IDF 向量
        self.vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = self._compute_tf(tokens)
            
            vector = np.zeros(len(self.vocab))
            for token, tf_val in tf.items():
                if token in self.vocab:
                    vector[self.vocab[token]] = tf_val * self.idf.get(token, 1)
            
            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            
            self.vectors.append(vector)
        
        self.vectors = np.array(self.vectors)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """检索相关文档"""
        if not self.documents:
            return []
        
        # 构建查询向量
        query_tokens = self._tokenize(query)
        query_vector = np.zeros(len(self.vocab))
        
        for token in query_tokens:
            if token in self.vocab:
                tf = query_tokens.count(token) / len(query_tokens)
                query_vector[self.vocab[token]] = tf * self.idf.get(token, 1)
        
        # 归一化
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm
        
        # 计算余弦相似度
        similarities = np.dot(self.vectors, query_vector)
        
        # 返回 top_k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.documents[i], float(similarities[i])) for i in top_indices]


class FAISSRetriever:
    """FAISS 向量检索器（适合大数据）"""
    
    def __init__(self, dimension: int = 768):
        import faiss
        self.dimension = dimension
        self.documents: List[str] = []
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(
        self,
        documents: List[str],
        embeddings: np.ndarray
    ):
        """添加文档和向量"""
        self.documents = documents
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """向量最近邻搜索"""
        query_embedding = np.array(query_embedding).astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        return [
            (self.documents[i], float(distances[0][j]))
            for j, i in enumerate(indices[0])
            if i < len(self.documents)
        ]


def test_retriever():
    """测试检索器"""
    # 测试 TF-IDF
    print("测试 TF-IDF Retriever...")
    retriever = TFIDFRetriever()
    
    documents = [
        "RAG是检索增强生成技术，解决LLM知识过时问题",
        "LangChain是LLM应用开发框架，支持RAG",
        "FAISS是向量数据库，用于高效检索",
        "Embedding将文本转为向量，实现语义匹配",
        "Agent是智能体，能自主决策"
    ]
    
    retriever.add_documents(documents)
    
    results = retriever.retrieve("RAG是什么", top_k=3)
    print("检索结果:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")
    
    # 测试 FAISS
    print("\n测试 FAISS Retriever...")
    import faiss
    
    faiss_retriever = FAISSRetriever(dimension=8)
    
    # 模拟 embeddings
    np.random.seed(42)
    embeddings = np.random.rand(len(documents), 8).astype('float32')
    faiss_retriever.add_embeddings(documents, embeddings)
    
    query = np.random.rand(8).astype('float32')
    results = faiss_retriever.search(query, top_k=3)
    print("FAISS 检索结果:")
    for doc, dist in results:
        print(f"  [{dist:.3f}] {doc}")


if __name__ == "__main__":
    test_retriever()
