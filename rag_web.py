import uuid

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = (5, 180)


st.set_page_config(page_title="油气安全知识库 Agent", page_icon="📘")
st.title("油气安全知识库 Agent")
st.caption("网页由 Streamlit 提供；问题通过 FastAPI 交给 RAG Agent。")
st.link_button("打开 FastAPI 接口文档", f"{API_BASE_URL}/docs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def clear_local_conversation():
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


with st.sidebar:
    st.subheader("对话与缓存")

    if st.button("清空对话", use_container_width=True):
        old_session_id = st.session_state.session_id
        try:
            requests.delete(
                f"{API_BASE_URL}/sessions/{old_session_id}",
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            # 后端暂时没启动时，换一个新会话 ID，避免接续旧记忆。
            pass
        clear_local_conversation()
        st.rerun()

    if st.button("清空程序缓存", use_container_width=True):
        try:
            response = requests.post(
                f"{API_BASE_URL}/cache/clear",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            old_session_id = st.session_state.session_id
            requests.delete(
                f"{API_BASE_URL}/sessions/{old_session_id}",
                timeout=REQUEST_TIMEOUT,
            )
            clear_local_conversation()
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"无法清理 FastAPI 缓存：{exc}")

    st.caption("清空程序缓存不会删除 Milvus 中的 PDF 索引。")
    st.caption(f"FastAPI 地址：{API_BASE_URL}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("查看 Agent 检索到的原文"):
                for source in message["sources"]:
                    st.markdown(source)


question = st.chat_input("例如：司钻岗位有哪些主要风险？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在通过 FastAPI 调用 Agent……"):
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": question,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()

            answer = payload["answer"]
            sources = payload.get("sources", [])
            st.markdown(answer)

            if sources:
                with st.expander("查看 Agent 检索到的原文"):
                    for source in sources:
                        st.markdown(source)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
            stats = payload.get("stats", {})
            st.caption(
                f"索引：{stats.get('collection', '未知')}；"
                f"去重后 {stats.get('chunks', '?')} 个文字块"
            )
        except requests.ConnectionError:
            st.error(
                "连不上 FastAPI。请先在另一个终端启动后端："
                "python -m uvicorn rag_api:app --host 127.0.0.1 --port 8000"
            )
        except requests.Timeout:
            st.error("请求超时。首次载入知识库可能较慢，请稍后重试。")
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"FastAPI 返回错误：{detail}")
        except requests.RequestException as exc:
            st.error(f"请求 FastAPI 时遇到问题：{exc}")
