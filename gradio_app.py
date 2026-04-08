import gradio as gr
import os
from typing import List, Optional, Any

def resolve_uploaded_path(uploaded_item: Any) -> Optional[str]:
    if uploaded_item is None:
        return None
    if isinstance(uploaded_item, str):
        return uploaded_item
    if isinstance(uploaded_item, dict):
        return uploaded_item.get("tmp_path") or uploaded_item.get("name")
    if hasattr(uploaded_item, "name"):
        return uploaded_item.name
    return None

def answer_with_files(files: List[Any], question: str, use_openai: bool, chunk_size: int, chunk_overlap: int) -> tuple[str, str]:
    if not files:
        return "请上传至少一个文档或图片文件。", ""
    if not question:
        return "请填写问题文本。", ""

    paths = []
    for item in files:
        resolved = resolve_uploaded_path(item)
        if resolved:
            paths.append(resolved)

    # 简化模式：显示文件信息
    answer = f"收到 {len(paths)} 个文件: {', '.join([os.path.basename(p) for p in paths])}。问题: {question}"
    sources = f"文件路径:\n" + "\n".join(paths)
    return answer, sources

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="教育文档智能问答") as demo:
        gr.Markdown("""
        # 教育场景文档智能问答系统
        上传 `txt` / `pdf` / `docx` / `pptx` 或 `图片`（支持 `png` / `jpg` / `jpeg` / `tif` / `tiff` / `bmp`），图片将自动进行试卷矫正和 OCR 识别。
        支持本地模型和 OpenAI 生成式问答。
        """)

        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="上传文档或图片", 
                    file_count="multiple", 
                    type="filepath",
                    file_types=[".txt", ".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
                )
                question_input = gr.Textbox(label="问题", placeholder="请输入你的问题，例如：这份文档讲了什么？", lines=2)
                use_openai_checkbox = gr.Checkbox(label="优先使用 OpenAI", value=False)
                chunk_size = gr.Slider(label="切分块大小", minimum=200, maximum=2000, step=100, value=800)
                chunk_overlap = gr.Slider(label="切分块重叠长度", minimum=0, maximum=400, step=50, value=100)
                submit_btn = gr.Button("开始问答")

            with gr.Column():
                answer_output = gr.Textbox(label="回答", lines=10)
                sources_output = gr.Textbox(label="参考片段", lines=12)

        submit_btn.click(
            answer_with_files,
            inputs=[file_input, question_input, use_openai_checkbox, chunk_size, chunk_overlap],
            outputs=[answer_output, sources_output],
        )

    return demo

if __name__ == "__main__":
    create_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
