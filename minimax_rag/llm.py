#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax API LLM 封装
提供同步和流式调用
"""

import time
import requests
from typing import Iterator, Optional


class MiniMaxLLM:
    """MiniMax API 调用封装"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.5",
        base_url: str = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    def invoke(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """同步调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            result = response.json()
            
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                raise Exception(result["error"])
            else:
                return str(result)
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式调用（返回迭代器）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=30
            )
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        # 解析 SSE 数据
                        # 这里简化处理，实际需要更复杂的解析
                        yield data_str
                        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """支持直接调用"""
        return self.invoke(prompt, **kwargs)


def test_llm():
    """测试 LLM 调用"""
    import os
    
    api_key = os.environ.get("MINIMAX_API_KEY", "your-api-key")
    llm = MiniMaxLLM(api_key)
    
    # 测试同步调用
    print("测试同步调用...")
    response = llm.invoke("用一句话解释RAG", temperature=0.7)
    print(f"响应: {response}")
    
    # 测试 Token 效率
    print("\n测试 Token 效率...")
    efficient_prompt = "解释RAG：1.定义 2.原理 3.应用（50字以内）"
    response = llm.invoke(efficient_prompt, max_tokens=100)
    print(f"高效响应: {response}")


if __name__ == "__main__":
    test_llm()
