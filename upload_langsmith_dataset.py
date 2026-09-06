import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


DATASET_PATH = Path(__file__).parent / "datasets" / "langsmith_dataset.json"
DATASET_NAME = "inflearn-streamlit-tax-qa"


def main():
    load_dotenv()
    legacy_env_file = Path("/Users/jjune/rag_LLM/.env")
    if legacy_env_file.exists():
        load_dotenv(legacy_env_file)

    if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError(
            "LANGSMITH_API_KEY 또는 LANGCHAIN_API_KEY가 .env에 설정되어 있지 않습니다."
        )

    examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"기존 Dataset 사용: {DATASET_NAME}")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="소득세 RAG 챗봇 평가 데이터셋",
        )
        print(f"Dataset 생성: {DATASET_NAME}")

    client.create_examples(
        inputs=[example["inputs"] for example in examples],
        outputs=[{"answer": example["expected_output"]} for example in examples],
        dataset_id=dataset.id,
    )
    print(f"업로드 완료: {len(examples)}개 예시")


if __name__ == "__main__":
    main()