# Day 02: 智能客服路由 Agent

目标：构建一个可控的客服路由 Agent，让模型只负责判断用户意图，业务流程由 LangGraph 的条件边执行。

JD 对标：

- AI 应用架构设计
- 可控性提升
- LLM 意图识别
- 图工作流编排

## 核心设计

State 是图中流转的字典，至少包含：

```python
{
    "user_input": "用户原始问题",
    "category": "refund 或 technical_support"
}
```

本示例额外加入：

```python
{
    "result": "最终客服回复",
    "classify_reason": "模型分类原因"
}
```

节点设计：

- `classifier_node`: 调用模型，把用户输入分类成 `refund` 或 `technical_support`
- `refund_node`: 处理退款、退货、取消订单类问题
- `technical_support_node`: 处理登录、报错、设备、使用方法类问题

条件边：

```text
classifier_node
    ├── category == refund              -> refund_node
    └── category == technical_support   -> technical_support_node
```

## 运行

安装依赖：

```powershell
cd d:\program\learning\agent\day02
pip install -r requirements.txt
```

设置 API Key：

```powershell
$env:OPENAI_API_KEY="你的 key"
```

运行：

```powershell
python customer_service_router.py
```

测试退款问题：

```text
我买错了商品，想申请退款
```

测试技术支持问题：

```text
我登录时一直提示验证码错误，怎么办？
```

## 为什么更可控

如果完全让模型生成客服回复，流程会比较不可控。

这个版本把职责拆开：

- 模型负责分类
- LangGraph 负责路由
- 业务 Node 负责确定性处理

这样可以清楚知道每个请求进入了哪个分支，也方便后续加入人工审核、日志记录、权限校验或工单系统。
