from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv("/Users/jjune/rag_LLM/.env")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

database = PineconeVectorStore.from_existing_index(
    index_name="tax-index-v2",
    embedding=embedding
)

retriever = database.as_retriever(
    search_kwargs={"k": 7}
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

dictionary = "사람을 나타내는 표현 → 거주자"

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """사용자의 질문에서 아래 사전의 의미와 같은 표현이 있을 때만
사전의 값으로 변경하세요.

사전:
{dictionary}

사전과 관련 없는 단어이거나 의미가 다르면 절대 변경하지 마세요.
인사말이나 불필요한 표현은 제거하고, 검색에 사용할 핵심 질문만 출력하세요.
변경 결과인 질문만 출력하세요.

예시:
질문: 연봉 5천만원인 직장인의 소득세는 얼마인가요?
답변: 연봉 5천만원인 거주자의 소득세는 얼마인가요?

질문: 연봉 5천만원인 피자의 소득세는 얼마인가요?
답변: 연봉 5천만원인 피자의 소득세는 얼마인가요."""
    ),
    ("human", "{question}")
])

rewrite_chain = (
    rewrite_prompt.partial(dictionary=dictionary)
    | llm
    | StrOutputParser()
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 한국 소득세 전문가입니다.

Context에 포함된 정보를 사용해서 답변하세요.
Context에 답이 없으면 '제공된 문서에서 확인할 수 없습니다.'라고 답변하세요.
확인되지 않은 세율이나 금액을 임의로 만들지 마세요.
계산이 필요한 경우 계산 과정을 간단히 설명하세요.
문서에 있는 공식을 사용해 계산하되, 부족한 조건은 가정으로 생각하고 대답하세요.


Context:
{context}"""
    ),
    ("human", "{question}")
])

answer_chain = answer_prompt | llm | StrOutputParser()


def get_ai_message(user_question):
    rewritten_question = rewrite_chain.invoke({"question": user_question})
    documents = retriever.invoke(rewritten_question)
    context = "\n\n".join(document.page_content for document in documents)
    answer = answer_chain.invoke({
        "context": context,
        "question": rewritten_question,
    })
    return answer, rewritten_question, documents