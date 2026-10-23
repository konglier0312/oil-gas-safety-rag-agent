# 油气安全手册 RAG Agent

这是一个基于油气安全手册的问答练习项目。网页由 Streamlit 提供，FastAPI 接收问题并调用 LangChain Agent；Agent 使用检索工具从 Milvus 查询 PDF 内容，再由 DeepSeek 根据检索片段生成带页码的回答。项目使用 BGE-M3 生成文本向量，并通过 SiliconFlow 调用 Embedding 服务。

## 当前功能

- 读取 PDF、按文本块切分，并根据文本哈希去重。
- 首次运行时生成向量并写入 Milvus；后续运行连接已有集合。
- Agent 可调用安全手册检索工具，并返回检索到的原文与页码。
- 支持多轮对话、清空当前会话和清空程序内缓存。
- 提供 FastAPI 接口和 Streamlit 聊天页面。

对话记录保存在 FastAPI 进程内，服务重启后清空。清理程序缓存不会删除 Milvus 中的向量集合。

## 准备环境

建议使用 Python 3.11。还需要一个可访问的 Milvus 服务，默认地址为 `http://localhost:19530`。

在项目目录打开 PowerShell，依次执行：

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入 SiliconFlow 和 DeepSeek 的 API Key。不要把 `.env` 或真实密钥提交到 GitHub。

## 添加 PDF

本仓库不包含原始手册文件。下载《陆上石油天然气钻井司钻岗位》PDF 后，将其命名为 `oil_gas_safety.pdf` 并放在项目根目录。该文件已加入 `.gitignore`，不会被 Git 上传。

## 启动

在第一个终端启动 API：

```powershell
python -m uvicorn rag_api:app --host 127.0.0.1 --port 8000 --reload
```

在第二个终端启动网页：

```powershell
python -m streamlit run rag_web.py
```

打开终端显示的 Streamlit 地址开始提问。FastAPI 文档页通常是 <http://127.0.0.1:8000/docs>，健康检查接口是 <http://127.0.0.1:8000/health>。

## 主要接口

- `POST /chat`：发送 `session_id` 和 `message`，返回回答、来源及索引信息。
- `DELETE /sessions/{session_id}`：清空指定会话。
- `POST /cache/clear`：清空进程内 Agent 缓存，不删除 Milvus 向量。

## 项目文件

- `rag_core.py`：PDF 处理、去重、向量检索和 Agent。
- `rag_api.py`：FastAPI 服务和会话管理。
- `rag_web.py`：Streamlit 聊天界面。
- `rag项目.py`：命令行问答入口。

## 已知限制

- 当前只针对单本 PDF 做练习，未包含 OCR、用户权限或持久化会话存储。
- 向量集合名和 Milvus 地址在代码中配置；需要调整时修改 `rag_core.py`。
- 检索效果取决于文档质量、切分方式和相关度阈值，仍需用问题集持续评估。

