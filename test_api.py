#!/usr/bin/env python3
"""
简单测试脚本
"""

from api import QAResult, create_qa_system

def test_basic():
    """测试基本功能"""
    print("测试 QAResult...")
    result = QAResult("测试回答", [], "测试问题", 0)
    print("✓ QAResult 创建成功")
    print(f"  结果: {result.to_dict()}")

    print("\n测试 DocumentQASystem 创建...")
    system = create_qa_system()
    print("✓ DocumentQASystem 创建成功")
    print(f"  配置: use_openai={system.use_openai}, chunk_size={system.chunk_size}")

    print("\n测试文档信息...")
    info = system.get_document_info()
    print(f"✓ 文档信息: {info}")

    print("\n所有基本测试通过！")

if __name__ == "__main__":
    test_basic()