import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DAY06_DIR = ROOT_DIR / "day06"
sys.path.insert(0, str(DAY06_DIR))

import fallback_rag_graph  # noqa: E402


DEFAULT_QUESTIONS = [
    "退钱一般多久到账？",
    "验证码一直错误怎么办？",
    "Plus 会员每个月有几张免邮券？",
    "企业发票需要填写哪些信息？",
    "定制商品可以无理由退款吗？",
]


JUDGE_SYSTEM_PROMPT = """你是一个无情、严格、可复现的 RAG 质量裁判。

你的任务不是回答用户问题，而是评估一个 RAG Agent 已经生成的最终答案质量。

你会收到三类信息：
1. 用户原始提问 original_question
2. Agent 检索到的上下文 retrieved_context
3. Agent 最终回答 final_answer

你必须根据 retrieved_context 判断 final_answer 的质量。
不要使用你自己的外部知识，不要因为答案听起来合理就给高分。
如果 final_answer 中出现 retrieved_context 没有支撑的事实、数字、承诺、流程、例外条件，必须判定为幻觉风险。

评分维度：

维度 A：幻觉程度 hallucination_score，1-5 分。
- 5 分：答案完全由 retrieved_context 支撑，没有任何额外事实或夸张承诺。
- 4 分：答案基本由 retrieved_context 支撑，只有非常轻微的概括，不影响事实准确性。
- 3 分：答案大体正确，但包含一些 context 中没有明确出现的推断或泛化。
- 2 分：答案混入明显未被 context 支撑的信息，存在较高幻觉风险。
- 1 分：答案大量编造，或与 context 明显冲突。

维度 B：相关性 relevance_score，1-5 分。
- 5 分：答案直接、完整、具体地回答了用户问题。
- 4 分：答案回答了核心问题，但有少量不必要信息或细节不足。
- 3 分：答案部分回答了问题，但遗漏关键条件、时间、范围或操作步骤。
- 2 分：答案只沾边，主要在复述 context，没有解决用户核心问题。
- 1 分：答案没有回答用户问题，或答非所问。

总分 total_score：
- 取 hallucination_score 和 relevance_score 的平均值，保留 1 位小数。
- 如果 final_answer 与 retrieved_context 冲突，总分最高不能超过 2.0。
- 如果 final_answer 没有正面回答问题，总分最高不能超过 3.0。

请只输出合法 JSON，不要输出 Markdown，不要输出额外解释。

JSON 格式必须是：
{
  "hallucination_score": 1到5的整数,
  "relevance_score": 1到5的整数,
  "total_score": 数字,
  "hallucination_reason": "说明答案哪些内容是否被上下文支撑",
  "relevance_reason": "说明答案是否切中用户问题",
  "failure_modes": ["可为空数组；列出主要问题，如 hallucination, incomplete_answer, irrelevant_context"],
  "actionable_suggestion": "给 RAG 系统的具体改进建议"
}
"""


def build_retrieved_context(documents: list[dict[str, Any]]) -> str:
    chunks = []
    for doc in documents:
        chunks.append(
            f"[{doc.get('id')} | {doc.get('title')} | score={doc.get('score')}]\n"
            f"{doc.get('content')}"
        )
    return "\n\n".join(chunks)


def run_day06_trajectory(question: str) -> dict[str, Any]:
    final_state = fallback_rag_graph.run(question)
    documents = final_state.get("documents", [])

    return {
        "question": question,
        "rewritten_question": final_state.get("rewritten_question", ""),
        "retrieved_documents": documents,
        "retrieved_context": build_retrieved_context(documents),
        "answer": final_state.get("answer", ""),
        "is_relevant": final_state.get("is_relevant"),
        "grade_reason": final_state.get("grade_reason", ""),
        "is_grounded": final_state.get("is_grounded"),
        "hallucination_reason": final_state.get("hallucination_reason", ""),
        "is_answer_useful": final_state.get("is_answer_useful"),
        "answer_grade_reason": final_state.get("answer_grade_reason", ""),
    }


def judge_trajectory(client: OpenAI, trajectory: dict[str, Any], judge_model: str) -> dict[str, Any]:
    payload = {
        "original_question": trajectory["question"],
        "rewritten_question": trajectory.get("rewritten_question", ""),
        "retrieved_context": trajectory["retrieved_context"],
        "final_answer": trajectory["answer"],
    }

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question",
        "rewritten_question",
        "answer",
        "doc_titles",
        "doc_scores",
        "day06_is_relevant",
        "day06_grade_reason",
        "day06_is_grounded",
        "day06_hallucination_reason",
        "day06_is_answer_useful",
        "day06_answer_grade_reason",
        "judge_hallucination_score",
        "judge_relevance_score",
        "judge_total_score",
        "judge_hallucination_reason",
        "judge_relevance_reason",
        "judge_failure_modes",
        "judge_actionable_suggestion",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(trajectory: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    docs = trajectory.get("retrieved_documents", [])
    return {
        "question": trajectory["question"],
        "rewritten_question": trajectory.get("rewritten_question", ""),
        "answer": trajectory.get("answer", ""),
        "doc_titles": " | ".join(doc.get("title", "") for doc in docs),
        "doc_scores": " | ".join(str(doc.get("score", "")) for doc in docs),
        "day06_is_relevant": trajectory.get("is_relevant"),
        "day06_grade_reason": trajectory.get("grade_reason", ""),
        "day06_is_grounded": trajectory.get("is_grounded"),
        "day06_hallucination_reason": trajectory.get("hallucination_reason", ""),
        "day06_is_answer_useful": trajectory.get("is_answer_useful"),
        "day06_answer_grade_reason": trajectory.get("answer_grade_reason", ""),
        "judge_hallucination_score": judgement.get("hallucination_score"),
        "judge_relevance_score": judgement.get("relevance_score"),
        "judge_total_score": judgement.get("total_score"),
        "judge_hallucination_reason": judgement.get("hallucination_reason", ""),
        "judge_relevance_reason": judgement.get("relevance_reason", ""),
        "judge_failure_modes": json.dumps(judgement.get("failure_modes", []), ensure_ascii=False),
        "judge_actionable_suggestion": judgement.get("actionable_suggestion", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Day07: offline RAG judge evaluation")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4.1"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--trajectories-json", default="")
    parser.add_argument("--skip-rag-run", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for Day07 judge evaluation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectories_path = output_dir / f"trajectories_{timestamp}.json"
    judgements_path = output_dir / f"judgements_{timestamp}.json"
    csv_path = output_dir / f"rag_judge_scores_{timestamp}.csv"

    if args.skip_rag_run:
        if not args.trajectories_json:
            raise ValueError("--skip-rag-run requires --trajectories-json")
        trajectories = load_json(Path(args.trajectories_json))
    else:
        trajectories = []
        for index, question in enumerate(DEFAULT_QUESTIONS, start=1):
            print(f"\n=== Running Day06 trajectory {index}/5 ===")
            print(f"Question: {question}")
            trajectories.append(run_day06_trajectory(question))
        save_json(trajectories_path, trajectories)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judgements = []
    rows = []

    for index, trajectory in enumerate(trajectories, start=1):
        print(f"\n=== Judging trajectory {index}/{len(trajectories)} ===")
        judgement = judge_trajectory(client, trajectory, args.judge_model)
        judgements.append(judgement)
        rows.append(flatten_result(trajectory, judgement))

    save_json(judgements_path, judgements)
    export_csv(csv_path, rows)

    print("\n=== Exported ===")
    if not args.skip_rag_run:
        print(f"Trajectories: {trajectories_path}")
    print(f"Judgements: {judgements_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
