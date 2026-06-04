# Day 03: Persistence (持久化) 与 MemorySaver

LangGraph 运行完一次图（Graph）后，它的状态（State）就会被丢弃。这就好比一个失忆的 Agent，每次你跟它说话，它都当你是第一次见面。**Persistence（持久化）机制就是为了解决这个问题而生的。**

## 跨轮对话的记忆 (Multi-turn Conversation)

**核心概念：`thread_id`**。当你配置了 `MemorySaver` 后，每次调用图时，你都需要传入一个 `thread_id`（线程 ID）。

**工作原理**：Agent 就像建了一个以 `thread_id` 命名的文件夹（称为 Checkpoint）。每一次运行结束时，图的最终 `State` 都会被作为快照（Checkpoint）保存下来。当用户带着同一个 `thread_id` 再次发起请求时，LangGraph 会自动把上一次保存的快照加载回来，作为本次运行的初始状态。这样，Agent 就“记住”了之前的上下文。

























目标：在 Day02 的客服路由 Agent 上加入 LangGraph checkpoint 记忆、中断点和人工授权恢复。

JD 对标：

- 长短期记忆系统 Memory
- 复杂工程架构
- Human-in-the-loop
- 可恢复工作流

## 新增能力

Day03 仍然保留 Day02 的核心图结构：

```text
START
  ↓
classifier
  ↓
条件边 route_by_category
  ├── refund_handler
  └── technical_support_handler
  ↓
END
```

在此基础上新增：

- `conversation_history`：短期记忆，记录同一个 `thread_id` 下的多轮对话
- `user_profile`：长期记忆的简化版，记录最近意图、订单号、退款次数等
- `MemorySaver`：基于内存的 LangGraph checkpointer
- `SqliteSaver`：可选的 SQLite checkpoint，把记忆写入本地文件
- `interrupt_before=["refund_handler"]`：退款 Node 执行前暂停，等待人工输入 `yes`

## 运行

安装依赖：

```powershell
cd d:\program\learning\agent\day03
pip install -r requirements.txt
```

有 OpenAI Key 时，模型负责分类：

```powershell
$env:OPENAI_API_KEY="你的 key"
python customer_service_router_memory.py
```

没有 OpenAI Key 时，脚本会自动使用本地关键词分类器，方便先验证 MemorySaver 和中断恢复。

也可以显式使用本地分类器：

```powershell
$env:USE_LOCAL_CLASSIFIER="1"
python customer_service_router_memory.py
```

## 验证 thread_id 记忆

默认 demo 会模拟同一个用户分两次提问：

```text
Turn 1: 我昨天买错了一个耳机，订单号 A10086，想申请退款。
Turn 2: 那这个大概要多久？
```

第二轮问题没有直接说“退款”，也没有订单号，但 Agent 会读取同一个 `thread_id` 下的历史 State：

```python
conversation_history
user_profile
```

因此可以把“多久到账”理解成上一轮退款问题的延续。

## 人工授权中断点

脚本编译图时使用：

```python
workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["refund_handler"],
)
```

当流程即将进入 `refund_handler` 时，LangGraph 会暂停。

控制台输入：

```text
yes
```

之后代码会用同一个 `thread_id` 调用：

```python
app.invoke(None, config=config)
```

从暂停点继续执行退款 Node。

## SQLite 记忆

默认使用内存版：

```powershell
python customer_service_router_memory.py
```

切换到 SQLite：

```powershell
python customer_service_router_memory.py --sqlite
```

指定记忆文件：

```powershell
python customer_service_router_memory.py --sqlite --sqlite-path day03_memory.sqlite
```

使用相同 `thread_id`，SQLite checkpoint 可以在下一次运行时继续读取之前保存的 State。

## 关闭中断点

```powershell
python customer_service_router_memory.py --no-interrupt
```

## 交互模式

```powershell
python customer_service_router_memory.py --interactive --thread-id user-1001
```

输入：

```text
exit
```

退出。
