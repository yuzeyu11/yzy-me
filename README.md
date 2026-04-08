# yzy-me

面向教育场景的文档智能问答系统，支持文档加载与智能分块、向量检索、生成式问答、以及图像预处理接口预留。

## 核心功能

- 📄 文档加载与智能分块：支持 `txt` / `pdf` / `docx` / `pptx`，图片 OCR（支持 `png` / `jpg` / `jpeg` / `tif` / `tiff` / `bmp`）
- 🔍 向量检索：基于 Chroma + `sentence-transformers`
- 🤖 生成式问答：支持 OpenAI 或本地模型回退
- 🖼️ 图像预处理模块：集成试卷矫正算法（透视矫正、旋转矫正）

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 使用

### 1. 本地命令行问答

```bash
python app.py --docs example.pdf example.txt --question "这份文档主要讲了什么？"
```

### 2. 开启 OpenAI 生成模型

```bash
export OPENAI_API_KEY="你的 OpenAI Key"
python app.py --docs example.pdf --question "请总结文档内容。" --use-openai
```

### 3. 启动 Gradio Web UI

```bash
python gradio_app.py
```

打开浏览器访问 `http://127.0.0.1:7860`。

### 4. 图片 OCR 直接加载

Gradio 界面支持直接上传图片文件，系统会自动进行图像预处理和 OCR 提取文本作为检索文档。

### 5. Python API 调用

```python
from api import DocumentQASystem

# 创建问答系统
qa_system = DocumentQASystem(use_openai=False)

# 加载文档
qa_system.load_documents(["document.pdf", "notes.txt"])

# 问答
result = qa_system.ask("这份文档讲了什么？")
print(result.answer)
```

### 6. REST API 服务

```bash
# 启动REST API服务
python rest_api.py
```

API端点：
- `GET /health` - 健康检查
- `POST /load` - 加载文档
- `POST /ask` - 问答
- `GET /info` - 获取系统信息
- `POST /clear` - 清除索引

### 7. 快速问答函数

```python
from api import ask_documents

result = ask_documents(
    file_paths=["doc.pdf"],
    question="总结文档内容",
    use_openai=True
)
print(result.answer)
```

## 文件结构

- `app.py`：主入口脚本，负责加载文档、构建向量索引并执行问答
- `src/document_loader.py`：文档加载与智能分块
- `src/retriever.py`：Chroma 向量检索封装
- `src/qa.py`：生成式问答与本地模型/OpenAI 接入
- `src/image_preprocessing.py`：图像预处理与 OCR 预留接口
- `api.py`：封装的问答系统API类，提供简洁的Python接口
- `rest_api.py`：REST API服务，提供HTTP接口
- `gradio_app.py`：Gradio Web UI界面
- `example.py`：使用示例脚本
- `requirements.txt`：依赖列表

## 扩展建议

- 已集成试卷矫正算法（透视矫正、旋转矫正）在 `src/image_preprocessing.py`
- 已补充 `docx` / `pptx` 文档格式支持在 `src/document_loader.py`
- 已添加 Gradio 可视化问答界面
- 已封装Python API接口（`api.py`）和REST API服务（`rest_api.py`）
- 考虑添加更多 OCR 语言支持或云端 OCR 服务集成
- 优化向量检索，支持更多嵌入模型或混合检索

