from rag_core import (
    CHAT_HISTORY_LIMIT,
    extract_agent_answer,
    load_rag_components,
)


def main():
    _, _, agent, stats = load_rag_components()
    print(f"{stats['index_status']}：{stats['collection']}")
    print(
        f"去重后 {stats['chunks']} 个文字块，"
        f"本次加载去掉 {stats['duplicates_removed']} 个重复块。"
    )
    print("可以连续追问；输入 /clear 清空对话记忆，输入 q 退出。")

    messages = []
    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("已退出问答。")
            break
        if question.lower() == "/clear":
            messages.clear()
            print("对话记忆已清空。")
            continue
        if not question:
            continue

        messages.append({"role": "user", "content": question})
        result = agent.invoke(
            {"messages": messages[-CHAT_HISTORY_LIMIT:]}
        )
        answer, _ = extract_agent_answer(result)
        messages.append({"role": "assistant", "content": answer})
        print("\n回答：")
        print(answer)


if __name__ == "__main__":
    main()
