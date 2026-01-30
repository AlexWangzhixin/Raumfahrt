#!/usr/bin/env python3
"""
Codex API测试脚本
测试是否可以使用Codex模型生成代码
"""

import requests
import json
import os


def test_codex_api():
    """测试Codex API调用"""
    print("=== 测试Codex API调用 ===")
    
    # API端点
    url = "https://api.openai.com/v1/chat/completions"
    
    # 使用用户之前提供的API密钥
    api_key = "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"
    
    # 请求数据 - 使用gpt-4o作为代码生成模型（Codex的替代品）
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a code assistant. Generate clean, efficient Python code."
            },
            {
                "role": "user",
                "content": "Write a function to calculate the factorial of a number using recursion."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # 尝试发送请求
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("发送Codex API请求中...")
        print("API端点:", url)
        print("请求模型:", payload["model"])
        print("请求内容:", payload["messages"][1]["content"])
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"\nAPI响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 解析响应
            response_data = response.json()
            
            print("\n=== API响应内容 ===")
            print(f"模型: {response_data['model']}")
            print(f"使用tokens: {response_data['usage']['total_tokens']}")
            print("\n生成的代码:")
            print(response_data['choices'][0]['message']['content'])
            
            print("\n✓ Codex API调用成功！")
            return True
        else:
            print(f"\n✗ API调用失败: {response.text}")
            return False
        
    except Exception as e:
        print(f"✗ API调用测试失败: {str(e)}")
        return False


def test_codex_for_project():
    """测试Codex为项目生成代码"""
    print("\n=== 测试Codex为项目生成代码 ===")
    
    # API端点
    url = "https://api.openai.com/v1/chat/completions"
    
    # 使用用户之前提供的API密钥
    api_key = "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"
    
    # 请求数据 - 为月球车项目生成代码
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a code assistant for a lunar rover navigation project. Generate clean, efficient Python code that follows the project's coding style."
            },
            {
                "role": "user",
                "content": "Write a function to calculate the distance between two points on the lunar surface, considering the moon's curvature. Use the haversine formula."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    # 尝试发送请求
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("发送Codex API请求中...")
        print("请求内容:", payload["messages"][1]["content"])
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"\nAPI响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 解析响应
            response_data = response.json()
            
            print("\n=== API响应内容 ===")
            print(f"模型: {response_data['model']}")
            print(f"使用tokens: {response_data['usage']['total_tokens']}")
            print("\n生成的代码:")
            print(response_data['choices'][0]['message']['content'])
            
            print("\n✓ Codex API项目代码生成成功！")
            return True
        else:
            print(f"\n✗ API调用失败: {response.text}")
            return False
        
    except Exception as e:
        print(f"✗ API调用测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始测试Codex API调用...\n")
    
    # 运行测试
    test1_result = test_codex_api()
    test2_result = test_codex_for_project()
    
    print("\n=== 测试总结 ===")
    print(f"Codex API基础测试: {'成功' if test1_result else '失败'}")
    print(f"Codex API项目代码生成测试: {'成功' if test2_result else '失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有Codex API测试通过！")
    else:
        print("\n⚠️  部分Codex API测试失败，请检查API配置。")
