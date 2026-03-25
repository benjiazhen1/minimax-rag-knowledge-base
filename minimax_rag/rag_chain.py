#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 问答链
整合检索器 + LLM 实现检索增强生成
"""

from typing import Optional, List, Dict


class RAGChain:
    """RAG 问答链"""
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_template: Optional[str] = None
    ):
        self.retriever = retriever
        self.llm = llm
        
        # 默认 Prompt 模板（Token 效率优化）
        self.prompt_template = prompt_template or """
基于背景知识回答问题。

要求：
1. 简洁准确，不超过100字
2. 只基于背景知识，不要编造

背景知识：
{context}

问题：{question}

答案："""
    
    def invoke(self, question: str, top_k: int = 3) -> Dict:
        """
        执行 RAG 问答
        
        Args:
            question: 用户问题
            top_k: 返回相关文档数量
            
        Returns:
            Dict: 包含 answer, sources, scores
        """
        # 1. 检索相关文档
        results = self.retriever.retrieve(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "抱歉，知识库中没有找到相关信息。",
                "sources": [],
                "scores": []
            }
        
        # 2. 构建上下文
        context_parts = []
        for doc, score in results:
            context_parts.append(f"- {doc}")
        
        context = "\n".join(context_parts)
        
        # 3. 构建 Prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # 4. 调用 LLM
        answer = self.llm.invoke(prompt, temperature=0.3)
        
        return {
            "answer": answer,
            "sources": [doc for doc, _ in results],
            "scores": [score for _, score in results]
        }
    
    def __call__(self, question: str, **kwargs) -> Dict:
        """支持直接调用"""
        return self.invoke(question, **kwargs)


def create_rag_chain(retriever, llm) -> RAGChain:
    """工厂函数：创建 RAG 链"""
    return RAGChain(retriever=retriever, llm=llm)


def test_rag_chain():
    """测试 RAG 链"""
    import os
    
    # 导入模块
    from minimax_rag import MiniMaxLLM, TFIDFRetriever, RAGChain
    
    # 初始化
    api_key = os.environ.get("MINIMAX_API_KEY", "your-api-key")
    llm = MiniMaxLLM(api_key)
    retriever = TFIDFRetriever()
    
    # 添加文档
    documents = [
        "RAG是检索增强生成技术，解决LLM知识过时问题",
        "LangChain是LLM应用开发框架，支持RAG",
        "FAISS是向量数据库，用于高效检索",
        "Embedding将文本转为向量，实现语义匹配",
        "Agent是智能体，能自主决策"
    ]
    retriever.add_documents(documents)
    
    # 创建 RAG 链
    rag = RAGChain(retriever, llm)
    
    # 测试问答
    print("测试 RAG 问答...")
    
    questions = [
        "什么是RAG？",
        "LangChain支持哪些功能？",
        "FAISS是什么？"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        result = rag.invoke(q)
        print(f"答案: {result['answer']}")
        print(f"来源: {result['sources'][0] if result['sources'] else '无'}")


if __name__ == "__main__":
    test_rag_chain()
