from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from config import answer_examples


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

history_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """사용자의 질문에서 아래 사전의 의미와 같은 표현이 있을 때만
사전의 값으로 변경하세요.

사전:
{dictionary}

사전과 관련 없는 단어이거나 의미가 다르면 절대 변경하지 마세요.
인사말이나 불필요한 표현은 제거하고, 검색에 사용할 핵심 질문만 출력하세요.
변경 결과인 검색 질문만 출력하세요.

이전 대화가 있다면 대화 맥락을 이용해 사용자의 질문을 완전한 질문으로
바꾸세요. 이전 대화가 없으면 현재 질문만 다듬으세요."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = (
    RunnableBranch(
        (
            lambda inputs: not inputs.get("chat_history"),
            RunnableLambda(lambda inputs: inputs["input"]) | retriever,
        ),
        history_rewrite_prompt.partial(dictionary=dictionary)
        | llm
        | StrOutputParser()
        | retriever,
    )
)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{answer}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=answer_examples,
    example_prompt=example_prompt,
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 한국 소득세 전문가입니다.

Context에 포함된 정보를 사용해서 답변하세요.
Context에 답이 없으면 '제공된 문서에서 확인할 수 없습니다.'라고 답변하세요.
확인되지 않은 세율이나 금액을 임의로 만들지 마세요.
계산이 필요한 경우 계산 과정을 간단히 설명하세요.
문서에 있는 공식을 사용해 계산하세요.
공제표는 금액이 속한 구간의 경계부터 확인한 뒤 해당 구간의 공식만 적용하세요.
예를 들어 총급여액 5,000만원은 4,500만원 초과 1억원 이하 구간이므로
근로소득공제는 1,200만원 + (5,000만원 - 4,500만원) × 5%입니다.
부양가족 등 질문에 없는 조건은 1인 근로자, 부양가족 없음으로 가정하고
그 가정을 답변에 명시하세요. 1인 근로자는 본인 기본공제 150만원을 적용하세요.
법 조항을 언급해도 좋아요.


Context:
{context}"""
    ),
    few_shot_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

answer_chain = answer_prompt | llm | StrOutputParser()

calculation_queries = [
    "제47조 근로소득공제 총급여액 4천500만원 1천200만원 5퍼센트",
    "거주자 본인 150만원 기본공제",
    "제55조 1,400만원 5,000만원 6퍼센트 15퍼센트",
]


def _retrieve_documents(search_question, initial_documents=None):
    documents = list(initial_documents or retriever.invoke(search_question))
    for calculation_query in calculation_queries:
        documents.extend(database.similarity_search(calculation_query, k=1))

    unique_documents = []
    seen_contents = set()
    for document in documents:
        if document.page_content not in seen_contents:
            unique_documents.append(document)
            seen_contents.add(document.page_content)
    return unique_documents


def _to_chat_messages(message_list):
    chat_history = []
    for message in message_list:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "ai":
            chat_history.append(AIMessage(content=message["content"]))
    return chat_history


def get_ai_message(user_question, message_list):
    chat_history = _to_chat_messages(message_list)
    documents = history_aware_retriever.invoke({
        "input": user_question,
        "chat_history": chat_history,
    })
    documents = _retrieve_documents(user_question, documents)
    context = "\n\n".join(document.page_content for document in documents)
    ai_response = answer_chain.stream({
        "context": context,
        "input": user_question,
        "chat_history": chat_history,
    })
    return ai_response, user_question, documents