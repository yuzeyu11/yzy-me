import argparse
import os
from pathlib import Path
from typing import List, Optional

from src.document_loader import load_documents
from src.qa import DocumentQA, QAEngine
from src.retriever import ChromaRetriever


def build_index(paths: List[str], persist_dir: str, chunk_size: int, chunk_overlap: int) -> ChromaRetriever:
    documents = load_documents(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not documents:
        raise ValueError("没有加载到任何文档，请检查输入路径。")

    retriever = ChromaRetriever(persist_dir=persist_dir)
    retriever.reset()
    texts = [item["content"] for item in documents]
    metadatas = [item["metadata"] for item in documents]
    ids = [f"doc_{idx}" for idx in range(len(texts))]
    retriever.add_documents(documents=texts, metadatas=metadatas, ids=ids)
    return retriever


def run_question(question: str, retriever: ChromaRetriever, use_openai: bool) -> dict:
    qa_engine = QAEngine()
    if use_openai and not qa_engine.openai_api_key:
        raise EnvironmentError("未检测到 OPENAI_API_KEY，无法使用 OpenAI 模型。")

    qa = DocumentQA(retriever, qa_engine)
    return qa.answer(question)


def ask_question(question: str, retriever: ChromaRetriever, use_openai: bool) -> None:
    result = run_question(question, retriever, use_openai)
    print("\n=== 回答 ===")
    print(result["answer"])
    print("\n=== 参考片段 ===")
    for hit in result["sources"]:
        print(f"- 来源: {hit['metadata'].get('source')}  距离: {hit['distance']:.4f}")
        print(hit["document"][:400].replace("\n", " "))
        print("---")


def main() -> None:
    parser = argparse.ArgumentParser(description="教育场景文档智能问答系统")
    parser.add_argument("--docs", nargs="+", help="待加载的文档路径，支持 txt/pdf/图片 OCR 预留", required=True)
    parser.add_argument("--question", help="问答问题文本", required=True)
    parser.add_argument("--persist-dir", default="./chroma_db", help="Chroma 向量库存储目录")
    parser.add_argument("--chunk-size", type=int, default=800, help="文档切分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="文档切分块重叠长度")
    parser.add_argument("--use-openai", action="store_true", help="优先使用 OpenAI 生成模型")
    args = parser.parse_args()

    retriever = build_index(args.docs, args.persist_dir, args.chunk_size, args.chunk_overlap)
    ask_question(args.question, retriever, args.use_openai)


if __name__ == "__main__":
    main()
