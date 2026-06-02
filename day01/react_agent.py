import os
import re
import ast
import operator
from openai import OpenAI


# System Prompt 是这个 Agent 的“行为说明书”。
#
# ReAct = Reason + Act:
# - Reason: 模型先用 Thought 思考下一步
# - Act: 如果需要外部信息或计算，就用 Action 请求调用工具
# - Observe: 程序真正调用本地函数，并把结果作为 Observation 喂回模型
# - Final: 模型拿到足够信息后输出 Final Answer，循环结束
#
# 这里没有使用 LangChain 的 AgentExecutor、Tool、Memory 等封装，
# 而是用字符串协议 + Python 循环把整个过程显式写出来。
SYSTEM_PROMPT = """You are a tiny ReAct agent.

You must answer by following this loop format:

Thought: think about what to do next
Action: tool_name(tool_input)

Available tools:
- get_weather(city): get a fake local weather report for a city
- calculate(math_expr): calculate a simple arithmetic expression

Rules:
- If you need a tool, output exactly one Action line.
- After you receive an Observation, continue the loop.
- When you know the answer, output:
Final Answer: your answer
"""


# 创建 OpenAI 客户端。
#
# 运行前需要在环境变量里设置 OPENAI_API_KEY：
# PowerShell:
#   $env:OPENAI_API_KEY="你的 key"
#
# 这里没有把 key 写死在代码里，是为了避免泄露。
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_weather(city: str) -> str:
    """一个假的本地天气工具。

    真实项目里，这里可以换成：
    - 调用天气 API
    - 查询数据库
    - 读取本地文件
    - 调用公司内部服务

    对 Agent 来说，工具就是普通 Python 函数。
    模型不会直接执行函数，它只会输出 Action 文本；
    真正执行函数的是下面的 Python 循环。
    """
    weather = {
        "beijing": "Beijing is sunny, 26 C.",
        "shanghai": "Shanghai is cloudy, 24 C.",
        "shenzhen": "Shenzhen has light rain, 28 C.",
        "hangzhou": "Hangzhou is breezy, 23 C.",
    }

    # 做一点简单清洗，让 " Shanghai "、"shanghai" 都能匹配。
    key = city.strip().lower()
    return weather.get(key, f"No local weather data for {city}.")


def calculate(math_expr: str) -> str:
    """安全地计算简单四则运算表达式。

    注意：不要直接 eval(model_output)。
    因为模型输出是文本，里面可能出现任意 Python 代码。

    这里用 ast.parse 把表达式解析成语法树，只允许：
    - 数字
    - 加减乘除
    - 幂运算
    - 负号

    这样既能演示“计算工具”，又避免执行危险代码。
    """

    # 把 AST 运算符节点映射到真正的 Python 运算函数。
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def visit(node):
        """递归遍历 AST 节点，只处理白名单里的表达式类型。"""

        # ast.Expression 是 mode="eval" 解析出来的根节点。
        if isinstance(node, ast.Expression):
            return visit(node.body)

        # 允许普通数字，比如 1、3.14。
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # 允许二元运算，比如 1 + 2、3 * 4。
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](visit(node.left), visit(node.right))

        # 允许一元运算，比如 -5。
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](visit(node.operand))

        # 其他任何东西都拒绝，比如函数调用、变量名、属性访问等。
        raise ValueError(f"Unsupported expression: {math_expr}")

    # mode="eval" 表示这里只解析“一个表达式”，而不是完整 Python 程序。
    tree = ast.parse(math_expr, mode="eval")
    return str(visit(tree))


# 工具注册表。
#
# key 是模型在 Action 里可以写的工具名；
# value 是真正会被 Python 调用的本地函数。
#
# 例如模型输出：
#   Action: calculate(1 + 2)
#
# 程序会解析出：
#   tool_name = "calculate"
#   tool_input = "1 + 2"
#
# 然后执行：
#   TOOLS["calculate"]("1 + 2")
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def call_llm(messages):
    """调用大模型。

    messages 是完整对话历史，形如：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "Thought: ... Action: ..."},
        {"role": "user", "content": "Observation: 工具返回结果"},
        ...
    ]

    ReAct Agent 的“记忆”在这个极简版本里就是 messages 列表。
    每一轮都会把模型回复和工具观察结果追加进去。
    """
    response = client.chat.completions.create(
        # 可以通过环境变量切换模型：
        #   $env:OPENAI_MODEL="gpt-4.1-mini"
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=messages,
        # temperature=0 让输出更稳定，更适合学习和调试协议格式。
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str):
    """从模型回复中解析 Action 行。

    期待模型输出类似：
      Thought: I need to calculate this.
      Action: calculate(12 * (3 + 4))

    这个函数会返回：
      ("calculate", "12 * (3 + 4)")

    如果没有找到 Action，就返回：
      (None, None)

    这是一个教学用的轻量解析器。
    生产环境里通常会使用 JSON 格式、函数调用接口，或者更严格的 parser。
    """

    # re.DOTALL 允许参数里出现换行。
    # (\w+) 捕获工具名，(.*) 捕获括号里的参数。
    match = re.search(r"Action:\s*(\w+)\((.*)\)", text, re.DOTALL)
    if not match:
        return None, None

    tool_name = match.group(1)
    tool_input = match.group(2).strip()

    # 允许模型写 Action: get_weather("shanghai")
    # 也允许模型写 Action: get_weather(shanghai)
    # 这里去掉最外层的单引号或双引号。
    tool_input = tool_input.strip("\"'")
    return tool_name, tool_input


def run_agent(question: str, max_steps: int = 8) -> str:
    """运行 ReAct 循环，直到模型给出 Final Answer。

    核心流程：
    1. 把 System Prompt 和用户问题放入 messages
    2. 调用模型，让模型输出 Thought / Action 或 Final Answer
    3. 如果是 Final Answer，直接结束
    4. 如果是 Action，解析工具名和参数
    5. Python 调用本地工具函数
    6. 把工具结果包装成 Observation，再放回 messages
    7. 回到第 2 步

    max_steps 是保险丝，避免模型一直不输出最终答案导致死循环。
    """

    # 初始上下文：系统规则 + 用户问题。
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Agent 的核心就是这个循环。
    # 每一轮模型要么请求调用一个工具，要么给出最终答案。
    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        # 让模型基于当前完整上下文决定下一步。
        reply = call_llm(messages)
        print(reply)

        # 把模型这次的回复追加到历史里。
        # 否则下一轮模型就不知道自己刚才想了什么、请求了什么工具。
        messages.append({"role": "assistant", "content": reply})

        # 如果模型已经给出最终答案，Agent 循环结束。
        if "Final Answer:" in reply:
            return reply.split("Final Answer:", 1)[1].strip()

        # 否则尝试从回复里解析 Action。
        tool_name, tool_input = parse_action(reply)

        # 如果模型没按格式输出，或者工具名不存在，就把错误也作为 Observation 喂回去。
        # 这样模型下一轮有机会自我修正。
        if tool_name not in TOOLS:
            observation = f"Unknown or missing action. Available tools: {list(TOOLS)}"
        else:
            try:
                # 真正执行工具的是 Python，不是模型。
                observation = TOOLS[tool_name](tool_input)
            except Exception as exc:
                # 工具报错也不要让程序直接崩掉；
                # 把错误反馈给模型，让它根据 Observation 继续决策。
                observation = f"Tool error: {exc}"

        print(f"Observation: {observation}")

        # 关键一步：
        # 把工具返回值变成一条新的用户消息 Observation。
        #
        # 下一轮模型看到 Observation 后，就能继续推理：
        # - 如果信息够了，输出 Final Answer
        # - 如果不够，再输出新的 Action
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 如果超过最大步数还没有 Final Answer，就停止。
    return "Reached max steps without a final answer."


if __name__ == "__main__":
    # 命令行入口。
    # 运行：
    #   python react_agent.py
    #
    # 然后输入问题，例如：
    #   What is the weather in Shanghai?
    #   What is 12 * (3 + 4)?
    question = input("Question: ").strip()
    answer = run_agent(question)
    print(f"\nFinal: {answer}")
