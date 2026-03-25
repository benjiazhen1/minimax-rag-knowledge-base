#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库示例
演示如何用 MiniMax API + RAG 构建智能知识库
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minimax_rag import MiniMaxLLM, TFIDFRetriever, RAGChain


def main():
    """主函数"""
    
    # 1. 初始化
    print("=" * 50)
    print("🚀 MiniMax RAG 知识库演示")
    print("=" * 50)
    
    # API Key（从环境变量读取）
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key or api_key == "your-api-key":
        print("⚠️ 请设置环境变量 MINIMAX_API_KEY")
        print("   export MINIMAX_API_KEY='your-api-key'")
        return
    
    llm = MiniMaxLLM(api_key)
    retriever = TFIDFRetriever()
    
    # 2. 准备知识库文档
    print("\n📚 加载知识库文档...")
    
    documents = [
        # AI 基础
        "RAG是检索增强生成技术，通过检索外部知识来增强LLM的生成能力",
        "LLM是大语言模型，能够理解和生成自然语言",
        "Embedding是将文本转为向量的技术，用于语义匹配",
        
        # LangChain
        "LangChain是一个用于构建LLM应用的开发框架",
        "LangChain支持Python和JavaScript两种语言",
        "LangChain的主要组件包括：Model I/O、Retrieval、Chains、Agents",
        
        # 向量数据库
        "FAISS是Facebook开源的高效向量检索库",
        "FAISS支持多种索引类型：Flat、IVF、HNSW",
        "向量数据库用于存储和检索文本的向量表示",
        
        # Agent
        "Agent是智能体，能够自主决策和执行任务",
        "ReAct是一种Agent模式：Thought + Action + Observation",
        "Tool Call是Agent调用外部工具的能力",
    ]
    
    retriever.add_documents(documents)
    print(f"   已加载 {len(documents)} 条文档")
    
    # 3. 创建 RAG 链
    print("\n🔗 创建 RAG 链...")
    rag = RAGChain(retriever, llm)
    print("   RAG 链创建成功")
    
    # 4. 问答演示
    print("\n" + "=" * 50)
    print("💬 问答演示")
    print("=" * 50)
    
    questions = [
        "什么是RAG？",
        "LangChain支持哪些功能？",
        "FAISS是什么？",
        "Agent有哪些能力？",
    ]
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        
        result = rag.invoke(question)
        
        print(f"✅ 答案: {result['answer']}")
        if result['sources']:
            print(f"📖 来源: {result['sources'][0]}")
    
    # 5. Token 效率测试
    print("\n" + "=" * 50)
    print("⚡ Token 效率测试")
    print("=" * 50)
    
    print("\n低 Token 消耗（精准 Prompt）：")
    efficient_prompt = "解释RAG：1句50字内"
    response = llm.invoke(efficient_prompt, max_tokens=100)
    print(f"   Prompt: {efficient_prompt}")
    print(f"   Response: {response}")


if __name__ == "__main__":
    main()
