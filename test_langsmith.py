import argparse

from llm import get_ai_message


def main():
    parser = argparse.ArgumentParser(description="Run one RAG question with LangSmith tracing.")
    parser.add_argument(
        "question",
        nargs="?",
        default="연봉 5천만원인 거주자의 소득세는 얼마인가요?",
    )
    args = parser.parse_args()

    response_stream, rewritten_question, documents = get_ai_message(args.question, [])
    response = "".join(response_stream)

    print(f"질문: {args.question}")
    print(f"검색 질문: {rewritten_question}")
    print(f"검색 문서 수: {len(documents)}")
    print(f"답변: {response}")


if __name__ == "__main__":
    main()