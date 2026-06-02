import json
import os
from typing import Literal, TypedDict

from openai import OpenAI
from langgraph.graph import END, START, StateGraph


Category = Literal["refund", "technical_support"]


class CustomerServiceState(TypedDict, total=False):
    """LangGraph 中流转的 State 字典结构。

    user_input:
        用户原始输入。

    category:
        分类器节点判断出的用户意图。
        本示例只允许两种可控分类：refund 或 technical_support。

    result:
        最终客服回复。

    classify_reason:
        分类原因，方便调试和审计。
    """

    user_input: str
    category: Category
    result: str
    classify_reason: str


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a customer service router.

Your job is to classify the user's message into exactly one category:

- refund: refund requests, return requests, order cancellation, payment reversal
- technical_support: login issues, app bugs, device problems, usage questions, error messages

Return only valid JSON:
{
  "category": "refund" or "technical_support",
  "reason": "short reason"
}
"""


def call_classifier_llm(user_input: str) -> dict:
    """调用模型做意图分类。

    注意这里的模型只负责“判断意图”，不直接决定业务流程怎么执行。
    业务流程由 LangGraph 的条件边控制，这样更可控、更容易审计。
    """

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


def classifier_node(state: CustomerServiceState) -> CustomerServiceState:
    """分类器 Node：读取 user_input，写入 category。

    这个节点是图里的入口节点。
    它把自然语言用户请求转成结构化分类结果，后续条件边会读取 category。
    """

    user_input = state["user_input"]
    data = call_classifier_llm(user_input)

    category = data.get("category")
    if category not in ("refund", "technical_support"):
        # 即使模型输出异常，也不要让流程失控。
        # 这里做一个保守兜底，把未知分类交给技术支持节点处理。
        category = "technical_support"

    return {
        "category": category,
        "classify_reason": data.get("reason", "No reason returned."),
    }


def refund_node(state: CustomerServiceState) -> CustomerServiceState:
    """退款处理 Node：处理退款、退货、取消订单类问题。"""

    user_input = state["user_input"]
    result = (
        "已识别为【退款/退货】问题。\n"
        "处理建议：请先核对订单号、支付状态和收货状态；如果符合退款规则，"
        "为用户创建退款工单，并告知预计 3-5 个工作日原路退回。\n"
        f"用户原始问题：{user_input}"
    )
    return {"result": result}


def technical_support_node(state: CustomerServiceState) -> CustomerServiceState:
    """技术支持 Node：处理登录、报错、设备、使用方法类问题。"""

    user_input = state["user_input"]
    result = (
        "已识别为【技术支持】问题。\n"
        "处理建议：请先收集设备型号、系统版本、错误截图或报错码；"
        "然后引导用户重试登录、清理缓存或升级到最新版本。\n"
        f"用户原始问题：{user_input}"
    )
    return {"result": result}


def route_by_category(state: CustomerServiceState) -> str:
    """条件边路由函数。

    LangGraph 会在 classifier_node 执行后调用这个函数。
    返回值必须能映射到 add_conditional_edges 里的 path_map。
    """

    return state["category"]


def build_graph():
    """构建并编译客服路由图。"""

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

    return workflow.compile()


def run(user_input: str) -> CustomerServiceState:
    """运行客服路由 Agent。"""

    app = build_graph()
    return app.invoke({"user_input": user_input})


if __name__ == "__main__":
    question = input("User: ").strip()
    final_state = run(question)

    print("\n--- Route Result ---")
    print(f"Category: {final_state['category']}")
    print(f"Reason: {final_state.get('classify_reason', '')}")
    print("\n--- Final Answer ---")
    print(final_state["result"])
