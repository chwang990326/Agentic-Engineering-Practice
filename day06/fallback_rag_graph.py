import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


GradeRoute = Literal["generate", "rewrite", "give_up"]
HallucinationRoute = Literal["check_answer", "regenerate", "give_up"]
AnswerRoute = Literal["finish", "rewrite", "give_up"]


class RagState(TypedDict, total=False):
    question: str
    original_question: str
    rewritten_question: str
    documents: list[dict]
    is_relevant: bool
    grade_reason: str
    answer: str
    retry_count: int
    generation_retry_count: int
    is_grounded: bool
    hallucination_reason: str
    is_answer_useful: bool
    answer_grade_reason: str


DOC_PATH = Path(__file__).with_name("business_policy.txt")
MAX_RETRIES = 2
MAX_GENERATION_RETRIES = 2


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese and English text with a tiny local tokenizer."""

    lower = text.lower()
    english = re.findall(r"[a-z0-9]+", lower)
    chinese = re.findall(r"[\u4e00-\u9fff]", lower)
    chinese_words = re.findall(r"[\u4e00-\u9fff]{2,}", lower)
    return english + chinese + chinese_words


def load_chunks() -> list[dict]:
    """Load the business document and split it into retrievable chunks."""

    text = DOC_PATH.read_text(encoding="utf-8")
    raw_chunks = [chunk.strip() for chunk in re.split(r"\n(?=## )", text) if chunk.strip()]
    chunks = []
    for index, chunk in enumerate(raw_chunks, start=1):
        title = chunk.splitlines()[0].lstrip("# ").strip()
        chunks.append(
            {
                "id": f"chunk-{index}",
                "title": title,
                "content": chunk,
            }
        )
    return chunks


def lexical_score(query: str, document: str) -> float:
    """A simple BM25-like score for a tiny demo corpus."""

    query_terms = tokenize(query)
    doc_terms = tokenize(document)
    if not query_terms or not doc_terms:
        return 0.0

    doc_len = len(doc_terms)
    term_counts = {term: doc_terms.count(term) for term in set(doc_terms)}
    score = 0.0
    for term in set(query_terms):
        tf = term_counts.get(term, 0)
        if tf == 0:
            continue
        score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_len / 120))

    return round(score, 4)


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Retrieve top-k chunks from the local business document."""

    scored = []
    for chunk in load_chunks():
        score = lexical_score(query, f"{chunk['title']}\n{chunk['content']}")
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def call_llm_json(system_prompt: str, user_payload: dict) -> dict:
    """Call OpenAI and require a JSON object response."""

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def call_llm_text(system_prompt: str, user_payload: dict) -> str:
    """Call OpenAI for answer generation."""

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def retrieve_node(state: RagState) -> RagState:
    """Node 1: retrieve documents for the current question."""

    question = state.get("rewritten_question") or state["question"]
    documents = retrieve(question)
    print(f"\n[retrieve] query = {question}")
    print(f"[retrieve] top scores = {[doc['score'] for doc in documents]}")
    return {"documents": documents}


def grade_documents_node(state: RagState) -> RagState:
    """Node 2: grade whether retrieved docs are relevant enough."""

    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    llm_grade = call_llm_json(
        "你是 RAG 检索结果评分器。只返回 JSON："
        '{"is_relevant": true/false, "reason": "简短原因"}',
        {
            "question": question,
            "documents": [
                {"title": doc["title"], "content": doc["content"], "score": doc["score"]}
                for doc in documents
            ],
        },
    )

    if not isinstance(llm_grade.get("is_relevant"), bool):
        raise ValueError(f"Invalid relevance grade: {llm_grade}")

    is_relevant = llm_grade["is_relevant"]
    reason = llm_grade.get("reason", "LLM graded relevance.")

    print(f"[grade] relevant = {is_relevant}; reason = {reason}")
    return {"is_relevant": is_relevant, "grade_reason": reason}


def route_after_grade(state: RagState) -> GradeRoute:
    """Conditional edge: generate, rewrite, or give up."""

    if state.get("is_relevant"):
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return "give_up"

    return "rewrite"


def rewrite_query_node(state: RagState) -> RagState:
    """Node 3: rewrite the query and loop back to retrieval."""

    current_question = state.get("rewritten_question") or state["question"]
    retry_count = state.get("retry_count", 0) + 1

    llm_rewrite = call_llm_json(
        "你是 RAG 查询改写器。只返回 JSON："
        '{"rewritten_question": "更适合检索业务文档的问题"}',
        {
            "original_question": state.get("original_question", state["question"]),
            "current_question": current_question,
            "grade_reason": state.get("grade_reason", ""),
            "retry_count": retry_count,
        },
    )

    rewritten = llm_rewrite.get("rewritten_question")
    if not rewritten:
        raise ValueError(f"Invalid rewritten query: {llm_rewrite}")

    print(f"[rewrite] retry_count = {retry_count}; rewritten = {rewritten}")
    return {
        "rewritten_question": rewritten,
        "retry_count": retry_count,
        "generation_retry_count": 0,
    }


def generate_answer_node(state: RagState) -> RagState:
    """Node 4: generate final answer from relevant documents."""

    question = state.get("original_question", state["question"])
    documents = state.get("documents", [])
    context = "\n\n".join(
        f"[{doc['title']} | score={doc['score']}]\n{doc['content']}" for doc in documents
    )

    generation_retry_count = state.get("generation_retry_count", 0) + 1
    llm_answer = call_llm_text(
        "你是企业知识库客服助手。只能基于给定文档回答；如果文档没有依据，要说明无法确认。",
        {"question": question, "context": context},
    )

    print(f"[generate] answer generated; attempt = {generation_retry_count}")
    return {"answer": llm_answer, "generation_retry_count": generation_retry_count}


def hallucination_grader_node(state: RagState) -> RagState:
    """Node 5: check whether the answer is fully grounded in retrieved context."""

    documents = state.get("documents", [])
    context = "\n\n".join(
        f"[{doc['title']} | score={doc['score']}]\n{doc['content']}" for doc in documents
    )
    grade = call_llm_json(
        "你是 RAG 幻觉检查器。判断 answer 是否完全能被 context 支撑。"
        "如果 answer 中有 context 没有依据的事实、数字、规则或承诺，就判定为不 grounded。"
        '只返回 JSON：{"is_grounded": true/false, "reason": "简短原因"}',
        {
            "question": state.get("original_question", state["question"]),
            "context": context,
            "answer": state["answer"],
        },
    )
    if not isinstance(grade.get("is_grounded"), bool):
        raise ValueError(f"Invalid hallucination grade: {grade}")

    is_grounded = grade["is_grounded"]
    reason = grade.get("reason", "LLM checked grounding.")
    print(f"[hallucination] grounded = {is_grounded}; reason = {reason}")
    return {"is_grounded": is_grounded, "hallucination_reason": reason}


def route_after_hallucination_check(state: RagState) -> HallucinationRoute:
    """Conditional edge after grounding check."""

    if state.get("is_grounded"):
        return "check_answer"

    if state.get("generation_retry_count", 0) >= MAX_GENERATION_RETRIES:
        return "give_up"

    return "regenerate"


def answer_grader_node(state: RagState) -> RagState:
    """Node 6: check whether the grounded answer actually answers the query."""

    grade = call_llm_json(
        "你是 RAG 答案有效性检查器。判断 answer 是否正面回答了 question 的核心诉求。"
        "只检查答案是否有用、具体、切题；不要检查事实来源。"
        '只返回 JSON：{"is_useful": true/false, "reason": "简短原因"}',
        {
            "question": state.get("original_question", state["question"]),
            "answer": state["answer"],
        },
    )
    if not isinstance(grade.get("is_useful"), bool):
        raise ValueError(f"Invalid answer grade: {grade}")

    is_useful = grade["is_useful"]
    reason = grade.get("reason", "LLM checked answer utility.")
    print(f"[answer_grade] useful = {is_useful}; reason = {reason}")
    return {"is_answer_useful": is_useful, "answer_grade_reason": reason}


def route_after_answer_check(state: RagState) -> AnswerRoute:
    """Conditional edge after answer utility check."""

    if state.get("is_answer_useful"):
        return "finish"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "give_up"

    return "rewrite"


def give_up_node(state: RagState) -> RagState:
    """Terminal node when retrieval remains irrelevant after retries."""

    answer = (
        "我尝试检索、改写、生成和质量检查后，仍没有得到足够可靠的答案。"
        "建议补充订单类型、业务场景或更具体的问题。"
    )
    print("[give_up] max retries reached")
    return {"answer": answer}


def build_graph():
    """Build the fallback RAG loop with LangGraph."""

    workflow = StateGraph(RagState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("hallucination_grader", hallucination_grader_node)
    workflow.add_node("answer_grader", answer_grader_node)
    workflow.add_node("give_up", give_up_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "give_up": "give_up",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", "hallucination_grader")
    workflow.add_conditional_edges(
        "hallucination_grader",
        route_after_hallucination_check,
        {
            "check_answer": "answer_grader",
            "regenerate": "generate_answer",
            "give_up": "give_up",
        },
    )
    workflow.add_conditional_edges(
        "answer_grader",
        route_after_answer_check,
        {
            "finish": END,
            "rewrite": "rewrite_query",
            "give_up": "give_up",
        },
    )
    workflow.add_edge("give_up", END)

    return workflow.compile()


def run(question: str) -> RagState:
    app = build_graph()
    return app.invoke(
        {
            "question": question,
            "original_question": question,
            "retry_count": 0,
            "generation_retry_count": 0,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day06 LangGraph fallback RAG")
    parser.add_argument(
        "--question",
        default="退钱一般多久到账？",
        help="用户问题",
    )
    args = parser.parse_args()

    final_state = run(args.question)
    print("\n=== Final Answer ===")
    print(final_state["answer"])
