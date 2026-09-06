import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from llm import get_ai_message


DATASET_NAME = "inflearn-streamlit-tax-qa"


def run_rag(inputs):
    question = inputs["question"]
    chat_history = inputs.get("chat_history", [])
    response_stream, _, _ = get_ai_message(question, chat_history)
    return {"answer": "".join(response_stream)}


def answer_is_not_empty(run, example):
    answer = (run.outputs or {}).get("answer", "").strip()
    return {
        "key": "answer_is_not_empty",
        "score": bool(answer),
        "comment": "답변이 생성되었습니다." if answer else "답변이 비어 있습니다.",
    }


def reference_answer_is_included(run, example):
    answer = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("answer", "")
    normalized_answer = "".join(answer.split())
    normalized_reference = "".join(reference.split())
    is_included = normalized_reference in normalized_answer
    return {
        "key": "reference_answer_is_included",
        "score": is_included,
        "comment": "기대 답변 전체가 포함되어 있습니다."
        if is_included
        else "표현이 달라 기대 답변 전체 일치는 실패했습니다.",
    }


def main():
    load_dotenv()
    legacy_env_file = Path("/Users/jjune/rag_LLM/.env")
    if legacy_env_file.exists():
        load_dotenv(legacy_env_file)

    if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError(
            "LANGSMITH_API_KEY 또는 LANGCHAIN_API_KEY가 설정되어 있지 않습니다."
        )

    results = evaluate(
        run_rag,
        data=DATASET_NAME,
        evaluators=[answer_is_not_empty, reference_answer_is_included],
        experiment_prefix="tax-rag-evaluation",
        description="소득세 RAG 챗봇 평가",
    )
    print(f"Experiment 실행 완료: {results.experiment_name}")


if __name__ == "__main__":
    main()