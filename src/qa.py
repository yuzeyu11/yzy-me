import os
from typing import Any, List, Optional

import torch
from transformers import pipeline

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None


class QAEngine:
    def __init__(self, local_model: str = "google/flan-t5-small", openai_model: str = "gpt-3.5-turbo"):
        self.local_model_name = local_model
        self.openai_model = openai_model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.local_generator = None
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key

    def _build_prompt(self, question: str, context: List[str]) -> str:
        context_text = "\n\n".join(context)
        return (
            "你是一个面向教育场景的智能问答助手，优先使用文档内容回答问题。"
            " 如果没有充足信息，请说明文档中未提及该内容。\n\n"
            f"参考文档内容：\n{context_text}\n\n"
            f"问题：{question}\n"
            "请给出简洁、准确的回答，必要时引用文档来源。"
        )

    def _load_local_generator(self):
        if self.local_generator is None:
            device = 0 if torch.cuda.is_available() else -1
            self.local_generator = pipeline(
                "text2text-generation",
                model=self.local_model_name,
                device=device,
                max_length=512,
                do_sample=False,
            )
        return self.local_generator

    def generate(self, question: str, context: List[str]) -> str:
        if self.openai_api_key and openai is not None:
            prompt = self._build_prompt(question, context)
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "你是一个教育场景下的文档智能问答助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()

        generator = self._load_local_generator()
        prompt = self._build_prompt(question, context)
        generated = generator(prompt, max_length=512)
        return generated[0]["generated_text"].strip()


class DocumentQA:
    def __init__(self, retriever: Any, qa_engine: QAEngine):
        self.retriever = retriever
        self.qa_engine = qa_engine

    def answer(self, question: str, top_k: int = 5) -> dict:
        hits = self.retriever.query(question, top_k=top_k)
        context = [hit["document"] for hit in hits]
        answer = self.qa_engine.generate(question, context)
        return {
            "answer": answer,
            "sources": hits,
        }
