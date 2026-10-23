import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "oil_gas_safety.pdf"
ENV_PATH = BASE_DIR / ".env"
MILVUS_URI = "http://localhost:19530"
# 新版本使用 COSINE；新集合名可避免误用旧的 L2 索引。旧集合会保留。
COLLECTION_NAME = "oil_gas_safety_agent_v3"
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHAT_HISTORY_LIMIT = 12  # 最近12条消息，约等于最近6轮问答
# COSINE 相关度映射到 0~1；该门槛是偏严格的初始值，需用问答样例校准。
MIN_VECTOR_RELEVANCE_SCORE = 0.88


def content_hash(text: str) -> str:
    """对文字做轻量规范化后计算 SHA-256，用作去重键和 Milvus 主键。"""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """按正文哈希去掉完全重复的块，并将哈希保存到元数据中。"""
    unique_documents = []
    seen_hashes = set()

    for document in documents:
        normalized = re.sub(r"\s+", " ", document.page_content).strip()
        if not normalized:
            continue

        digest = content_hash(normalized)
        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        metadata = dict(document.metadata)
        metadata["content_hash"] = digest
        unique_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )

    return unique_documents


def find_keyword_chunks(question: str, chunks: list[Document]) -> list[Document]:
    """用少量中文关键词补充向量检索，作为本项目的简单兜底。"""
    if any(word in question for word in ["风险类型", "主要风险", "主要安全风险"]):
        exact_matches = [
            chunk for chunk in chunks
            if "司钻岗位可能发生的主要风险类型有" in chunk.page_content
        ]
        return exact_matches[:3]

    normalized_question = question
    stop_words = [
        "可能由哪些情况造成", "可能由", "哪些情况", "有哪些", "主要原因",
        "这本手册", "本手册", "手册", "司钻岗位", "司钻", "岗位", "规定",
        "事故", "可能", "哪些", "情况", "造成", "原因", "主要", "怎么",
        "如何", "什么", "多少", "的", "是", "那", "这个", "怎么做",
    ]
    # 先删较长短语，避免先删“司钻”后残留“岗位”等泛词造成误命中。
    for word in sorted(stop_words, key=len, reverse=True):
        normalized_question = normalized_question.replace(word, "")
    normalized_question = "".join(
        char for char in normalized_question if "\u4e00" <= char <= "\u9fff"
    )

    if not normalized_question:
        return []

    query_terms = {
        normalized_question[index:index + 2]
        for index in range(len(normalized_question) - 1)
    }
    if len(normalized_question) == 1:
        query_terms = {normalized_question}

    minimum_score = 2 if len(query_terms) >= 2 else 1
    scored_chunks = []
    for chunk in chunks:
        score = sum(term in chunk.page_content for term in query_terms)
        if score >= minimum_score:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:3]]


def retrieve_documents(
    question: str,
    chunks: list[Document],
    vector_store: Milvus,
    k: int = 8,
) -> list[Document]:
    """合并明确关键词命中和通过相关度门槛的向量结果，并按正文哈希去重。"""
    keyword_results = find_keyword_chunks(question, chunks)
    scored_vector_results = vector_store.similarity_search_with_relevance_scores(
        question, k=k
    )
    vector_results = [
        document
        for document, score in scored_vector_results
        if score >= MIN_VECTOR_RELEVANCE_SCORE
    ]

    results = []
    seen_hashes = set()
    for document in keyword_results + vector_results:
        digest = document.metadata.get("content_hash") or content_hash(
            document.page_content
        )
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        results.append(document)

    return results[:5]


def format_sources(documents: list[Document]) -> str:
    if not documents:
        return "没有检索到相关资料。"

    sections = []
    for document in documents:
        page_number = document.metadata.get("page_label")
        if not page_number:
            page_number = document.metadata.get("page", 0) + 1
        source_name = Path(
            document.metadata.get("source", PDF_PATH.name)
        ).name

        sections.append(
            f"【来源：{source_name}，PDF第{page_number}页】\n"
            f"{document.page_content}"
        )
    return "\n\n".join(sections)


def extract_agent_answer(agent_result: dict) -> tuple[str, list[str]]:
    """从 Agent 执行结果中取最终回答和工具检索到的原文。"""
    messages = agent_result.get("messages", [])
    answer = ""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", []):
            content = message.content
            if isinstance(content, str):
                answer = content
            elif isinstance(content, list):
                answer = "\n".join(
                    part.get("text", str(part))
                    if isinstance(part, dict) else str(part)
                    for part in content
                )
            if answer:
                break

    sources = []
    seen_source_hashes = set()
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        content = message.content
        if not isinstance(content, str) or content == "没有检索到相关资料。":
            continue
        digest = content_hash(content)
        if digest not in seen_source_hashes:
            seen_source_hashes.add(digest)
            sources.append(content)

    if not answer:
        answer = "这次没有生成文字回答，请换一种问法再试一次。"

    # 模型已经判定缺少证据时，不展示未达相关度标准的候选原文，
    # 也不让模型附加“整本手册没有此内容”之类的推断。
    no_evidence_markers = (
        "当前检索到的资料中没有找到相关内容",
        "资料中没有找到",
        "没有检索到相关资料",
    )
    if any(marker in answer for marker in no_evidence_markers):
        return "当前检索到的资料中没有找到相关内容。", []

    return answer, sources


def load_rag_components():
    """读取 PDF、按哈希去重、连接/建立 Milvus 集合并创建 Agent。"""
    load_dotenv(ENV_PATH, override=True)

    required_keys = [
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise RuntimeError(
            "环境变量未配置：" + "、".join(missing_keys)
            + f"。请检查 {ENV_PATH}，不要把密钥写进代码。"
        )
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"找不到 PDF 文件：{PDF_PATH}")

    pages = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    # PDF第1页是封面，第2、3页是目录；这些页不进入知识库索引。
    chunks = [
        chunk for chunk in chunks
        if str(chunk.metadata.get("page_label")) not in {"1", "2", "3"}
    ]
    original_count = len(chunks)
    chunks = deduplicate_documents(chunks)
    deduplicated_count = original_count - len(chunks)

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL_NAME,
        openai_api_base=os.getenv("SILICONFLOW_BASE_URL"),
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
    )

    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(collection_name=COLLECTION_NAME):
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
        )
        index_status = "已连接到现有索引"
    else:
        # 哈希同时作为稳定的字符串主键；集合只在第一次运行时写入。
        vector_store = Milvus.from_texts(
            texts=[chunk.page_content for chunk in chunks],
            embedding=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.metadata["content_hash"] for chunk in chunks],
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            auto_id=False,
            index_params={
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {},
            },
        )
        index_status = "已创建新索引并写入资料"

    llm = init_chat_model(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )

    @tool
    def search_safety_manual(query: str) -> str:
        """检索油气安全手册。把用户问题写成完整、具体的问题后再搜索。"""
        documents = retrieve_documents(query, chunks, vector_store)
        return format_sources(documents)

    agent = create_agent(
        model=llm,
        tools=[search_safety_manual],
        system_prompt="""
你是油气安全手册问答 Agent。你可以调用 search_safety_manual 工具检索《高危行业一线岗位安全生产指导手册》。

回答手册中的事实、风险、原因、预防和处置问题前，必须先调用检索工具。
如果用户的问题依赖前文（例如“那怎么预防？”），结合最近对话把它补成一个独立、具体的问题，再调用工具。
只根据工具检索到且与问题直接相关的资料回答，不要用常识补写手册内容。
只回答用户当前明确询问的内容，不要主动扩展到其他风险类型或相关知识；需要展开时，等用户追问。
只有某个检索片段直接支持回答中的事实时，才引用该片段的来源文件和 PDF 页码；不得因为片段被检索出来就把它列为答案依据。
如果工具返回“没有检索到相关资料”，或返回的片段没有直接支持答案，只回答“当前检索到的资料中没有找到相关内容”，不要附上来源页码，也不要据此断言整本手册都没有这项内容。
""",
    )

    stats = {
        "chunks": len(chunks),
        "duplicates_removed": deduplicated_count,
        "index_status": index_status,
        "collection": COLLECTION_NAME,
    }
    return chunks, vector_store, agent, stats

