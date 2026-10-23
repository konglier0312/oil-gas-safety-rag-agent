from threading import Lock, RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import CHAT_HISTORY_LIMIT, extract_agent_answer, load_rag_components


app = FastAPI(
    title="油气安全知识库 Agent API",
    description="通过 RAG Agent 检索油气安全手册并回答问题。",
    version="1.0.0",
)

_components: tuple | None = None
_components_lock = RLock()
_conversations: dict[str, list[dict[str, str]]] = {}
_session_locks: dict[str, Lock] = {}
_sessions_lock = RLock()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    stats: dict[str, Any]


def get_components():
    global _components
    with _components_lock:
        if _components is None:
            _components = load_rag_components()
        return _components


def get_session_lock(session_id: str) -> Lock:
    with _sessions_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = Lock()
        return _session_locks[session_id]


@app.get("/health")
def health():
    with _components_lock:
        is_loaded = _components is not None
    return {
        "status": "ok",
        "components_loaded": is_loaded,
        "collection": "oil_gas_safety_agent_v1",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空。")

    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id 不能为空。")

    session_lock = get_session_lock(session_id)
    with session_lock:
        history = list(_conversations.get(session_id, []))
        history.append({"role": "user", "content": question})

        try:
            _, _, agent, stats = get_components()
            result = agent.invoke({"messages": history[-CHAT_HISTORY_LIMIT:]})
            answer, sources = extract_agent_answer(result)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Agent 处理失败：{exc}",
            ) from exc

        history.append({"role": "assistant", "content": answer})
        _conversations[session_id] = history[-CHAT_HISTORY_LIMIT:]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        stats=stats,
    )


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    session_lock = get_session_lock(session_id)
    with session_lock:
        _conversations.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}


@app.post("/cache/clear")
def clear_program_cache():
    global _components
    with _components_lock:
        _components = None
    return {
        "cleared": True,
        "milvus_index_deleted": False,
        "message": "程序内存中的 Agent 和连接缓存已清除；Milvus 持久索引仍保留。",
    }
