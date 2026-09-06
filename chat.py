import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# for message in st.session_state.message_list:
    # with st.chat_message(message["role"]):
    #     st.write(message["content"])

# print(f"before == {st.session_state.message_list}")
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
# print(f"after == {st.session_state.message_list}")

    with st.chat_message("ai"):
        with st.spinner("답변을 생성하는 중입니다..."):
            ai_message, rewritten_question, documents = get_ai_message(user_question)
        st.write(ai_message)
        with st.expander("답변 근거 확인"):
            st.caption(f"검색에 사용한 질문: {rewritten_question}")
            for document_index, document in enumerate(documents, start=1):
                st.markdown(f"**문서 {document_index}**")
                st.write(document.page_content)
                if document.metadata:
                    st.json(document.metadata)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})
print(f"after == {st.session_state.message_list}")


            