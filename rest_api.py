"""
文档智能问答系统 REST API
提供HTTP接口进行文档问答
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from typing import Dict, Any
import json

from api import DocumentQASystem, QAResult

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx', 'pptx', 'tif', 'tiff', 'bmp'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局问答系统实例
qa_system = None

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_qa_system() -> DocumentQASystem:
    """初始化问答系统"""
    global qa_system
    if qa_system is None:
        qa_system = DocumentQASystem(
            persist_dir="./api_db",
            use_openai=os.getenv("USE_OPENAI", "false").lower() == "true"
        )
    return qa_system

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "Document QA API",
        "version": "1.0.0"
    })

@app.route('/load', methods=['POST'])
def load_documents():
    """
    加载文档
    支持文件上传或URL路径
    """
    try:
        system = init_qa_system()

        file_paths = []

        # 处理上传的文件
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    file_paths.append(filepath)

        # 处理URL参数中的文件路径
        if 'file_paths' in request.json:
            paths = request.json['file_paths']
            if isinstance(paths, list):
                file_paths.extend(paths)
            else:
                file_paths.append(paths)

        if not file_paths:
            return jsonify({
                "error": "没有提供有效的文件",
                "allowed_extensions": list(ALLOWED_EXTENSIONS)
            }), 400

        # 加载文档
        success = system.load_documents(file_paths)

        if success:
            info = system.get_document_info()
            return jsonify({
                "message": "文档加载成功",
                "documents_loaded": info["documents"],
                "file_paths": file_paths
            })
        else:
            return jsonify({"error": "文档加载失败"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    问答接口
    """
    try:
        system = init_qa_system()

        if not system.is_loaded:
            return jsonify({"error": "请先加载文档"}), 400

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "请提供问题"}), 400

        question = data['question']
        top_k = data.get('top_k', 5)

        # 执行问答
        result = system.ask(question, top_k=top_k)

        return jsonify(result.to_dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def get_info():
    """获取系统信息"""
    try:
        system = init_qa_system()
        info = system.get_document_info()

        return jsonify({
            "system_info": info,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
            "use_openai": os.getenv("USE_OPENAI", "false").lower() == "true"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_index():
    """清除索引"""
    try:
        system = init_qa_system()
        success = system.clear_index()

        if success:
            return jsonify({"message": "索引已清除"})
        else:
            return jsonify({"error": "清除索引失败"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    """列出上传的文件"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                files.append({
                    "filename": filename,
                    "path": filepath,
                    "size": os.path.getsize(filepath)
                })

        return jsonify({"files": files})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("启动文档问答 API 服务...")
    print("支持的文件类型:", ALLOWED_EXTENSIONS)
    print("使用 OpenAI:", os.getenv("USE_OPENAI", "false"))

    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )