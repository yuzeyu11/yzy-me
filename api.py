"""
教育场景文档智能问答系统 API 接口
提供简洁的封装接口，支持文档加载、问答和结果返回
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class QAResult:
    """问答结果数据类"""
    answer: str
    sources: List[Dict]
    question: str
    total_sources: int

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "question": self.question,
            "total_sources": self.total_sources
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"Q: {self.question}\nA: {self.answer}\nSources: {self.total_sources}"


class DocumentQASystem:
    """文档智能问答系统封装类"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        use_openai: bool = False,
        openai_model: str = "gpt-3.5-turbo",
        local_model: str = "google/flan-t5-small"
    ):
        """
        初始化问答系统

        Args:
            persist_dir: 向量数据库存储目录
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠长度
            use_openai: 是否使用 OpenAI
            openai_model: OpenAI 模型名称
            local_model: 本地模型名称
        """
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.local_model = local_model

        # 延迟加载
        self._qa_engine = None
        self._retriever = None
        self.is_loaded = False

    @property
    def qa_engine(self):
        """延迟加载QA引擎"""
        if self._qa_engine is None:
            from src.qa import QAEngine
            self._qa_engine = QAEngine(
                local_model=self.local_model,
                openai_model=self.openai_model
            )
        return self._qa_engine

    @property
    def retriever(self):
        """延迟加载检索器"""
        if self._retriever is None:
            from src.retriever import ChromaRetriever
            self._retriever = ChromaRetriever(persist_dir=str(self.persist_dir))
        return self._retriever

    def load_documents(self, file_paths: Union[str, List[str]]) -> bool:
        """
        加载文档并构建向量索引

        Args:
            file_paths: 文档路径，可以是单个路径或路径列表

        Returns:
            bool: 是否成功加载
        """
        try:
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            # 检查文件是否存在
            valid_paths = []
            for path in file_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"警告: 文件不存在 - {path}")

            if not valid_paths:
                raise ValueError("没有找到有效的文档文件")

            # 加载文档
            from src.document_loader import load_documents
            documents = load_documents(
                valid_paths,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            if not documents:
                raise ValueError("文档加载失败，没有提取到内容")

            # 初始化检索器
            self.retriever.reset()

            # 添加文档到索引
            texts = [item["content"] for item in documents]
            metadatas = [item["metadata"] for item in documents]
            ids = [f"doc_{idx}" for idx in range(len(texts))]

            self.retriever.add_documents(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            self.is_loaded = True
            print(f"成功加载 {len(valid_paths)} 个文档，共 {len(texts)} 个分块")
            return True

        except Exception as e:
            print(f"文档加载失败: {str(e)}")
            self.is_loaded = False
            return False

    def ask(self, question: str, top_k: int = 5) -> QAResult:
        """
        执行问答

        Args:
            question: 问题文本
            top_k: 返回的参考文档数量

        Returns:
            QAResult: 问答结果
        """
        if not self.is_loaded:
            raise RuntimeError("请先加载文档")

        if not question.strip():
            raise ValueError("问题不能为空")

        try:
            # 检查 OpenAI 配置
            if self.use_openai and not self.qa_engine.openai_api_key:
                raise EnvironmentError("未检测到 OPENAI_API_KEY，无法使用 OpenAI 模型")

            # 执行问答
            from src.qa import DocumentQA
            qa = DocumentQA(self.retriever, self.qa_engine)
            result = qa.answer(question, top_k=top_k)

            return QAResult(
                answer=result["answer"],
                sources=result["sources"],
                question=question,
                total_sources=len(result["sources"])
            )

        except Exception as e:
            # 返回错误结果
            return QAResult(
                answer=f"问答失败: {str(e)}",
                sources=[],
                question=question,
                total_sources=0
            )

    def get_document_info(self) -> Dict:
        """
        获取已加载文档的信息

        Returns:
            Dict: 文档信息
        """
        if not self.is_loaded:
            return {"loaded": False, "documents": 0}

        try:
            # 查询集合中的文档数量
            collection = self.retriever.client.get_collection(self.retriever.collection_name)
            count = collection.count()

            return {
                "loaded": True,
                "documents": count,
                "persist_dir": str(self.persist_dir)
            }
        except Exception as e:
            return {
                "loaded": False,
                "error": str(e)
            }

    def clear_index(self) -> bool:
        """
        清除向量索引

        Returns:
            bool: 是否成功清除
        """
        try:
            if self._retriever:
                self._retriever.reset()
            self.is_loaded = False
            return True
        except Exception as e:
            print(f"清除索引失败: {str(e)}")
            return False


# 便捷函数
def create_qa_system(**kwargs) -> DocumentQASystem:
    """
    创建问答系统实例

    Returns:
        DocumentQASystem: 问答系统实例
    """
    return DocumentQASystem(**kwargs)


def ask_documents(
    file_paths: Union[str, List[str]],
    question: str,
    **kwargs
) -> QAResult:
    """
    快速问答函数：加载文档并回答问题

    Args:
        file_paths: 文档路径
        question: 问题
        **kwargs: 其他参数传递给 DocumentQASystem

    Returns:
        QAResult: 问答结果
    """
    system = create_qa_system(**kwargs)
    if system.load_documents(file_paths):
        return system.ask(question)
    else:
        return QAResult(
            answer="文档加载失败",
            sources=[],
            question=question,
            total_sources=0
        )


if __name__ == "__main__":
    # 示例用法
    system = create_qa_system()

    # 加载文档
    success = system.load_documents(["README.md"])
    if success:
        # 问答
        result = system.ask("这个项目是什么？")
        print(result)
        print("\n详细结果:")
        print(result.to_dict())
    else:
        print("文档加载失败")