import argparse
import os

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool


load_dotenv()


def build_agents() -> tuple[Agent, Agent, Agent]:
    """Define the three CrewAI agents used in this serial workflow."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    search_tool = SerperDevTool()

    researcher = Agent(
        role="Researcher / 研究员",
        goal=(
            "围绕用户给定主题进行联网搜索，提炼可信事实、趋势、案例、"
            "用户痛点和内容角度。"
        ),
        backstory=(
            "你是一名严谨的研究员，擅长把碎片化网页信息整理成结构化研究简报。"
            "你不会夸大结论；如果信息不确定，会明确标注风险。"
        ),
        tools=[search_tool],
        llm=model,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer / 爆款写手",
        goal=(
            "基于研究员的简报，写出有传播力、有标题钩子、有观点密度的中文文章草稿。"
        ),
        backstory=(
            "你是一名爆款写手，擅长把复杂主题写得清楚、有节奏、有读者代入感。"
            "你可以增强表达，但不能编造研究简报里没有支撑的事实。"
        ),
        llm=model,
        verbose=True,
        allow_delegation=False,
    )

    editor = Agent(
        role="Editor / 主编审核员",
        goal=(
            "审核文章草稿的事实风险、结构完整性、表达力度和发布安全性，"
            "输出可以发布的终稿。"
        ),
        backstory=(
            "你是一名主编审核员，既关心传播效果，也关心准确性、边界和品牌风险。"
            "你会删掉夸张、不实、空泛的表达，并保留真正有价值的观点。"
        ),
        llm=model,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, editor


def build_tasks(
    researcher: Agent,
    writer: Agent,
    editor: Agent,
) -> tuple[Task, Task, Task]:
    """Define three serial tasks and their explicit context dependencies."""

    research_task = Task(
        description=(
            "围绕主题「{topic}」进行联网搜索。\n"
            "请输出一份研究简报，必须包含：\n"
            "1. 主题背景\n"
            "2. 关键事实或趋势\n"
            "3. 可引用的案例或观察\n"
            "4. 适合写成爆款文章的角度\n"
            "5. 不确定性与风险提示\n"
        ),
        expected_output=(
            "一份 Markdown 研究简报，结构清晰，包含可供 Writer 使用的事实、观点和素材。"
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "读取上游 research_task 的研究简报，围绕主题「{topic}」写一篇中文爆款文章草稿。\n"
            "文章需要包含：\n"
            "1. 有吸引力的标题\n"
            "2. 能抓住读者的开头\n"
            "3. 三到五个核心观点\n"
            "4. 结合研究简报里的事实和案例\n"
            "5. 有行动启发的结尾\n"
        ),
        expected_output=(
            "一篇 Markdown 中文文章草稿，兼顾传播力和事实准确性。"
        ),
        agent=writer,
        context=[research_task],
    )

    editing_task = Task(
        description=(
            "读取 research_task 的研究简报和 writing_task 的文章草稿，进行主编审核。\n"
            "请完成：\n"
            "1. 检查事实是否有研究简报支撑\n"
            "2. 删除夸张或不可验证的表述\n"
            "3. 优化标题、结构和语言节奏\n"
            "4. 输出审核意见\n"
            "5. 输出最终可发布版本\n"
        ),
        expected_output=(
            "一份 Markdown 终稿，先给出主编审核意见，再给出最终可发布文章。"
        ),
        agent=editor,
        context=[research_task, writing_task],
    )

    return research_task, writing_task, editing_task


def build_crew() -> tuple[Crew, tuple[Task, Task, Task]]:
    """Build a sequential CrewAI crew."""

    researcher, writer, editor = build_agents()
    research_task, writing_task, editing_task = build_tasks(
        researcher=researcher,
        writer=writer,
        editor=editor,
    )

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew, (research_task, writing_task, editing_task)


def kickoff(topic: str):
    """One-click entrypoint: run Researcher -> Writer -> Editor."""

    crew, tasks = build_crew()
    result = crew.kickoff(inputs={"topic": topic})

    research_task, writing_task, editing_task = tasks
    print("\n\n================ Task Outputs ================")
    print("\n--- research_task output ---")
    print(research_task.output.raw if research_task.output else "[No output]")

    print("\n--- writing_task output ---")
    print(writing_task.output.raw if writing_task.output else "[No output]")

    print("\n--- editing_task output ---")
    print(editing_task.output.raw if editing_task.output else "[No output]")

    print("\n\n================ Crew Final Result ================")
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day04 CrewAI multi-agent workflow")
    parser.add_argument(
        "--topic",
        default="多 Agent 协作如何提升 AI 应用工程化能力",
        help="Researcher、Writer、Editor 要共同处理的主题",
    )
    args = parser.parse_args()

    kickoff(args.topic)
