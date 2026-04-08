#!/usr/bin/env python3
"""
文档问答系统使用示例
演示如何使用封装的API进行文档问答
"""

import os
from api import DocumentQASystem, ask_documents, QAResult

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 创建问答系统
    qa_system = DocumentQASystem(
        persist_dir="./example_db",
        use_openai=False  # 使用本地模型
    )

    # 加载文档（这里使用示例文件，你需要替换为实际文件）
    sample_files = ["README.md"]  # 替换为你的文档文件

    if qa_system.load_documents(sample_files):
        print("文档加载成功")

        # 问答
        question = "这个项目是什么？"
        result = qa_system.ask(question)

        print(f"\n问题: {result.question}")
        print(f"回答: {result.answer}")
        print(f"参考来源数量: {result.total_sources}")

        # 显示参考来源
        for i, source in enumerate(result.sources[:3]):  # 只显示前3个
            print(f"\n来源 {i+1}:")
            print(f"  文件: {source['metadata']['source']}")
            print(f"  相似度: {source['distance']:.4f}")
            print(f"  内容: {source['document'][:200]}...")

    else:
        print("文档加载失败")

def example_quick_ask():
    """快速问答示例"""
    print("\n=== 快速问答示例 ===")

    # 一行代码完成问答
    result = ask_documents(
        file_paths=["README.md"],  # 替换为你的文档
        question="这个系统支持哪些功能？",
        use_openai=False
    )

    print(f"问题: {result.question}")
    print(f"回答: {result.answer}")

def example_with_openai():
    """使用OpenAI示例"""
    print("\n=== OpenAI示例 ===")

    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        qa_system = DocumentQASystem(use_openai=True)

        if qa_system.load_documents(["README.md"]):
            result = qa_system.ask("请详细介绍这个项目")
            print(f"OpenAI回答: {result.answer}")
    else:
        print("未设置OPENAI_API_KEY，跳过OpenAI示例")

def example_batch_questions():
    """批量问答示例"""
    print("\n=== 批量问答示例 ===")

    qa_system = DocumentQASystem()

    if qa_system.load_documents(["README.md"]):
        questions = [
            "这个项目的主要功能是什么？",
            "支持哪些文档格式？",
            "如何安装和使用？"
        ]

        for question in questions:
            result = qa_system.ask(question)
            print(f"\nQ: {question}")
            print(f"A: {result.answer[:100]}...")

def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    qa_system = DocumentQASystem()

    # 尝试在未加载文档时问答
    try:
        result = qa_system.ask("测试问题")
    except RuntimeError as e:
        print(f"预期的错误: {e}")

    # 加载不存在的文件
    success = qa_system.load_documents(["nonexistent_file.txt"])
    print(f"加载不存在文件的结果: {success}")

def main():
    """主函数"""
    print("文档智能问答系统使用示例")
    print("=" * 50)

    # 检查是否有示例文件
    if not os.path.exists("README.md"):
        print("注意: 请确保有文档文件可以测试")
        print("当前目录文件:", os.listdir("."))
        return

    try:
        example_basic_usage()
        example_quick_ask()
        example_with_openai()
        example_batch_questions()
        example_error_handling()

        print("\n" + "=" * 50)
        print("所有示例完成！")

    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()