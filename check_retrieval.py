from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv("/Users/jjune/rag_LLM/.env")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

database = PineconeVectorStore.from_existing_index(
    index_name="tax-index-v2",
    embedding=embedding,
)

query = "연봉 5천만원인 거주자의 소득세는 얼마인가요?"
retrieved_docs = database.similarity_search_with_score(query, k=8)

print(f"질문: {query}")
print(f"검색 문서 수: {len(retrieved_docs)}")

for document_index, (document, score) in enumerate(retrieved_docs, start=1):
    print(f"\n--- 근거 {document_index} | score: {score} ---")
    print(document.page_content[:1000])
    print(f"metadata: {document.metadata}")
