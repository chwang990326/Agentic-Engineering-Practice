# Day 04: CrewAI 多 Agent 协作框架

## Agent：招募你的“领域专家” (Role)

在 crewAI 中，Agent 不是一个无所不知的通用聊天机器人，而是被深度定制的**专才（Specialist）**。 定义一个 Agent 的过程，本质上就是在写一份极度精确的“岗位说明书”。

**role** 先告诉模型“你是爆款写手”，模型会倾向于调动写作、标题、传播、结构表达相关能力；**goal** 再告诉它这次要干什么；**backstory** 则补充它的工作风格和约束。

**Role（角色头衔）**：这是赋予大语言模型（LLM）人设（Persona）的第一步。例如：`资深量化数据分析师` 或 `首席安全审计工程师`。它决定了模型在庞大知识库中优先检索哪一类专业术语和解决模式。

**Goal（北极星目标）**：这个 Agent 的终极目的是什么？它相当于 Agent 行动的方向标。例如：`编写最高效、无冗余的数据清洗脚本`。在遇到多目标冲突或歧义时，Agent 会以这个目标为基准做出取舍。

**Backstory（背景故事）**：**（🌟 最精髓的部分）** 这是 crewAI 提升模型表现的核心 Prompt 技巧。你不仅仅在告诉它“你是一个工程师”，而是要给它注入“灵魂”和“偏见”。

```
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
```

在多智能体系统中，最怕的就是 Agent 之间同质化。强烈的 **Backstory** 能够收敛 LLM 的注意力机制（Attention Mechanism），迫使它在特定任务上输出极其专业且深度的内容，过滤掉“正确的废话”。



## Task：定义不可妥协的“交付物” (Task)

如果说 Agent 是“谁来做”，那么 Task 就是“做什么”以及“做到什么程度”。 在多个 Agent 协同配合时，任务定义的边界越清晰，整个系统的出错率（幻觉）就越低。一个标准的 Task 必须包含以下三个核心要素：

**Description（任务描述）**：详细说明需要完成的具体工作，并提供必要的上下文信息。

**Agent（任务认领）**：明确将这个任务绑定给哪一位特定的专家（我们在上一步定义的 Role）。

**Expected Output（预期输出）**：**（🌟 协同的关键）** 这是连接不同 Task 之间的桥梁。在 crewAI 中，上一个任务的输出，往往会直接作为下一个任务的输入（Context）。因此，你必须在这里强定义数据格式。

```
research_task = Task(     
        description=(         //表示“做什么”
            "围绕主题「{topic}」进行联网搜索。\n"
            "请输出一份研究简报，必须包含：\n"
            "1. 主题背景\n"
            "2. 关键事实或趋势\n"
            "3. 可引用的案例或观察\n"
            "4. 适合写成爆款文章的角度\n"
            "5. 不确定性与风险提示\n"
        ),
        expected_output=(     //表示“做到什么程度，输出长什么样”
            "一份 Markdown 研究简报，结构清晰，包含可供 Writer 使用的事实、观点和素材。"
        ),
        agent=researcher,     //表示“谁来做”
    )
```



```
writing_task = Task(
    description="读取上游 research_task 的研究简报...",
    expected_output="一篇 Markdown 中文文章草稿...",
    agent=writer,
    context=[research_task],
)
```

**context=[research_task]** ：writing_task 依赖 research_task 的输出。