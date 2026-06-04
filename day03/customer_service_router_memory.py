import argparse
import json
import os
import re
import sqlite3
from typing import Any, Literal, TypedDict

from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


Category = Literal["refund", "technical_support"]


class CustomerServiceState(TypedDict, total=False):
    """客服路由 Agent 的全局 State。

    Day02 只有 user_input/category/result。
    Day03 在这个基础上加入两类记忆：

    conversation_history:
        短期记忆，记录当前 thread_id 下连续几轮对话。

    user_profile:
        长期记忆的简化版，保存用户画像、最近意图、订单号、累计次数等。
        如果 checkpointer 换成 SQLite，这些状态还能跨进程保存。
    """

    user_input: str
    category: Category
    result: str
    classify_reason: str
    conversation_history: list[dict[str, str]]
    user_profile: dict[str, Any]


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if os.environ.get("OPENAI_API_KEY") else None


CLASSIFIER_SYSTEM_PROMPT = """你是一个智能客服路由 Agent 的意图分类器。

你需要根据用户当前输入，并结合历史对话和用户画像，将用户意图分类为以下两类之一：

- refund：退款、退货、取消订单、支付退回、退款进度、多久到账
- technical_support：登录问题、软件报错、设备问题、使用方法、验证码问题

重要规则：
- 如果当前问题很短，例如“那什么时候到账？”、“刚才那个怎么处理？”，请结合历史对话判断。
- 只返回合法 JSON，不要输出 Markdown，不要输出解释性文本。

返回格式：
{
  "category": "refund" 或 "technical_support",
  "reason": "简短说明分类原因"
}
"""


def format_history(history: list[dict[str, str]], max_items: int = 8) -> str:
    """把短期记忆压成 prompt 可读文本。"""

    if not history:
        return "暂无历史对话。"

    recent = history[-max_items:]
    return "\n".join(f"{item['role']}: {item['content']}" for item in recent)


def extract_order_id(text: str) -> str | None:
    """从用户输入中提取一个非常简化的订单号。"""

    patterns = [
        r"订单号[:：\s]*([A-Za-z0-9-]+)",
        r"order\s*id[:：\s]*([A-Za-z0-9-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def local_classifier(
    user_input: str,
    conversation_history: list[dict[str, str]],
    user_profile: dict[str, Any],
) -> dict:
    """没有 OPENAI_API_KEY 时用于本地演示的分类器。

    真实运行时，有 API Key 会走 call_classifier_llm。
    这个 fallback 只是为了让 MemorySaver、thread_id、中断恢复可以离线验证。
    """

    refund_keywords = ["退款", "退货", "取消订单", "退回", "到账", "订单"]
    tech_keywords = ["登录", "验证码", "报错", "打不开", "闪退", "bug", "错误"]

    if any(keyword in user_input for keyword in refund_keywords):
        category = "refund"
        reason = "本地规则命中退款相关关键词。"
    elif any(keyword in user_input for keyword in tech_keywords):
        category = "technical_support"
        reason = "本地规则命中技术支持相关关键词。"
    elif user_profile.get("last_category"):
        category = user_profile["last_category"]
        reason = "当前问题较短，沿用同一 thread_id 中上一轮意图。"
    elif conversation_history:
        category = "technical_support"
        reason = "存在历史对话，但没有明确退款关键词，保守路由到技术支持。"
    else:
        category = "technical_support"
        reason = "没有明确意图，保守路由到技术支持。"

    return {"category": category, "reason": reason}


def call_classifier_llm(
    user_input: str,
    conversation_history: list[dict[str, str]],
    user_profile: dict[str, Any],
) -> dict:
    """调用模型做意图分类。

    Day03 的关键变化：
    分类器不只看 user_input，还会看到同一 thread_id 下保存的短期记忆
    conversation_history，以及简化长期记忆 user_profile。
    """

    if os.environ.get("USE_LOCAL_CLASSIFIER") == "1" or client is None:
        return local_classifier(user_input, conversation_history, user_profile)

    memory_context = {
        "conversation_history": format_history(conversation_history),
        "user_profile": user_profile,
        "current_user_input": user_input,
    }

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(memory_context, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


def update_user_profile(
    profile: dict[str, Any],
    user_input: str,
    category: Category,
) -> dict[str, Any]:
    """更新长期记忆的简化用户画像。"""

    updated = dict(profile)
    updated["last_category"] = category
    updated["last_user_input"] = user_input

    if category == "refund":
        updated["refund_request_count"] = int(updated.get("refund_request_count", 0)) + 1
        updated["last_refund_question"] = user_input
    else:
        updated["technical_support_count"] = int(updated.get("technical_support_count", 0)) + 1
        updated["last_technical_question"] = user_input

    order_id = extract_order_id(user_input)
    if order_id:
        updated["last_order_id"] = order_id

    return updated


def classifier_node(state: CustomerServiceState) -> CustomerServiceState:
    """分类器 Node：读取当前输入和记忆，写入 category。"""

    user_input = state["user_input"]
    conversation_history = list(state.get("conversation_history", []))
    user_profile = dict(state.get("user_profile", {}))

    data = call_classifier_llm(user_input, conversation_history, user_profile)

    category = data.get("category")
    if category not in ("refund", "technical_support"):
        category = user_profile.get("last_category", "technical_support")

    classify_reason = data.get("reason", "No reason returned.")
    updated_profile = update_user_profile(user_profile, user_input, category)
    updated_history = conversation_history + [
        {"role": "user", "content": user_input},
        {"role": "classifier", "content": f"category={category}; reason={classify_reason}"},
    ]

    return {
        "category": category,
        "classify_reason": classify_reason,
        "conversation_history": updated_history,
        "user_profile": updated_profile,
    }


def refund_node(state: CustomerServiceState) -> CustomerServiceState:
    """退款处理 Node：执行前可以通过 interrupt_before 暂停等待人工授权。"""

    user_input = state["user_input"]
    profile = state.get("user_profile", {})
    order_id = profile.get("last_order_id", "未提供")

    result = (
        "已识别为【退款/退货】问题。\n"
        f"关联订单号：{order_id}\n"
        "处理建议：请先核对订单号、支付状态和收货状态；如果符合退款规则，"
        "为用户创建退款工单，并告知预计 3-5 个工作日原路退回。\n"
        f"用户当前问题：{user_input}"
    )

    history = list(state.get("conversation_history", []))
    history.append({"role": "assistant", "content": result})
    return {"result": result, "conversation_history": history}


def technical_support_node(state: CustomerServiceState) -> CustomerServiceState:
    """技术支持 Node：处理登录、报错、设备、使用方法类问题。"""

    user_input = state["user_input"]
    result = (
        "已识别为【技术支持】问题。\n"
        "处理建议：请先收集设备型号、系统版本、错误截图或报错码；"
        "然后引导用户重试登录、清理缓存或升级到最新版本。\n"
        f"用户当前问题：{user_input}"
    )

    history = list(state.get("conversation_history", []))
    history.append({"role": "assistant", "content": result})
    return {"result": result, "conversation_history": history}


def route_by_category(state: CustomerServiceState) -> str:
    """条件边：根据分类结果动态选择下一个 Node。"""

    return state.get("category", "technical_support")


def make_checkpointer(use_sqlite: bool = False, sqlite_path: str = "day03_memory.sqlite"):
    """创建 LangGraph checkpointer。

    MemorySaver:
        基于内存，适合教学和单次运行。

    SqliteSaver:
        基于 SQLite，适合把 thread_id 的状态保存到本地文件。
    """

    if not use_sqlite:
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpoint 需要安装：pip install langgraph-checkpoint-sqlite"
        ) from exc

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer


def build_graph(checkpointer, interrupt_before_refund: bool = True):
    """构建并编译带记忆和中断点的客服路由图。"""

    workflow = StateGraph(CustomerServiceState)

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("refund_handler", refund_node)
    workflow.add_node("technical_support_handler", technical_support_node)

    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "refund": "refund_handler",
            "technical_support": "technical_support_handler",
        },
    )
    workflow.add_edge("refund_handler", END)
    workflow.add_edge("technical_support_handler", END)

    interrupt_before = ["refund_handler"] if interrupt_before_refund else None
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


def run_turn(app, user_input: str, thread_id: str) -> CustomerServiceState:
    """运行一轮用户输入。

    同一个 thread_id 会复用同一份 checkpoint。
    如果图停在 refund_handler 前，控制台输入 yes 后继续执行。
    """

    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"user_input": user_input}, config=config)
    snapshot = app.get_state(config)

    if "refund_handler" in snapshot.next:
        print("\n[Interrupt] 即将执行退款 Node，需要人工授权。")
        print(f"[Interrupt] 当前分类：{state.get('category')}")
        print(f"[Interrupt] 分类原因：{state.get('classify_reason')}")
        approved = input("输入 yes 继续执行退款处理，其他输入则暂停：").strip().lower()

        if approved == "yes":
            state = app.invoke(None, config=config)
        else:
            print("已暂停在退款 Node 前。下次可用同一个 thread_id 继续恢复。")

    return state


def print_state(state: CustomerServiceState) -> None:
    """打印本轮结果和记忆摘要。"""

    print("\n--- Route Result ---")
    print(f"Category: {state.get('category')}")
    print(f"Reason: {state.get('classify_reason')}")

    print("\n--- Final Answer ---")
    print(state.get("result", "[当前图仍处于暂停状态，还没有执行最终处理 Node]"))

    print("\n--- Memory Snapshot ---")
    print("user_profile:")
    print(json.dumps(state.get("user_profile", {}), ensure_ascii=False, indent=2))
    print(f"conversation_history length: {len(state.get('conversation_history', []))}")


def run_demo(app, thread_id: str) -> None:
    """模拟同一用户分两次提问，验证 Agent 记得上一轮上下文。"""

    questions = [
        "我昨天买错了一个耳机，订单号 A10086，想申请退款。",
        "那这个大概要多久？",
    ]

    for index, question in enumerate(questions, start=1):
        print(f"\n================ Turn {index} ================")
        print(f"User: {question}")
        state = run_turn(app, question, thread_id)
        print_state(state)


def run_interactive(app, thread_id: str) -> None:
    """交互式运行，输入 exit 退出。"""

    while True:
        question = input("\nUser: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        state = run_turn(app, question, thread_id)
        print_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day03: customer service router with memory")
    parser.add_argument("--thread-id", default="demo-user-001")
    parser.add_argument("--sqlite", action="store_true", help="使用 SQLite checkpointer")
    parser.add_argument("--sqlite-path", default="day03_memory.sqlite")
    parser.add_argument("--no-interrupt", action="store_true", help="关闭退款前人工授权中断点")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    args = parser.parse_args()

    checkpointer = make_checkpointer(
        use_sqlite=args.sqlite,
        sqlite_path=args.sqlite_path,
    )
    app = build_graph(
        checkpointer=checkpointer,
        interrupt_before_refund=not args.no_interrupt,
    )

    print(f"thread_id: {args.thread_id}")
    print(f"checkpointer: {'SQLite' if args.sqlite else 'MemorySaver'}")

    if args.interactive:
        run_interactive(app, args.thread_id)
    else:
        run_demo(app, args.thread_id)
