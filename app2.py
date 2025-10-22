from urllib import response
from requests import session
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# RAG 관련 import
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# API KEY 정보로드
load_dotenv()


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("대화내용을 기억하는 챗봇 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0)

    # 세션 ID 를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# RAG: worlds.txt 파일을 로드하고 벡터 저장소 생성
@st.cache_resource
def load_worlds_vectorstore():
    """worlds.txt 파일을 로드하고 FAISS 벡터 저장소를 생성합니다."""
    try:
        # 1. 문서 로드
        loader = TextLoader("worlds.txt", encoding="utf-8")
        documents = loader.load()

        # 2. 텍스트 분할 (월드별로 나누기)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # 3. 임베딩 생성 및 벡터 저장소 구축
        # 한국어 특화 SentenceTransformer 모델 사용
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        return vectorstore
    except Exception as e:
        st.error(f"벡터 저장소 로드 실패: {e}")
        return None


# RAG를 활용한 retriever 생성
def get_retriever():
    """벡터 저장소에서 retriever를 반환합니다."""
    vectorstore = load_worlds_vectorstore()
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 결과 반환
    return None


# 체인 생성
def create_chain(model_name="gpt-4.1-mini"):
    # retriever 가져오기
    retriever = get_retriever()

    # 프롬프트 정의 - RAG 컨텍스트 포함
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 ZEP 메타버스의 월드 추천 전문가입니다.
사용자의 감정이나 요구사항을 이해하고, 아래 제공된 월드 정보를 바탕으로 가장 적합한 월드를 추천해주세요.

# 월드 정보:
{context}

위 정보를 참고하여 사용자에게 친근하고 상세하게 월드를 소개하고 추천해주세요.
월드의 테마, 플레이 로직, 감정 키워드 등을 고려하여 답변하세요.""",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4.1-mini")

    # RAG 체인 생성
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # retriever를 사용하여 컨텍스트를 가져오는 체인
    def rag_chain_func(inputs):
        # retriever로 관련 문서 검색
        if retriever:
            # invoke 메서드 사용 (LangChain 최신 버전)
            docs = retriever.invoke(inputs["question"])
            context = format_docs(docs)
        else:
            context = "월드 정보를 로드할 수 없습니다."

        # 프롬프트에 컨텍스트 추가
        inputs["context"] = context
        return inputs

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history, rag_chain_func


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if "chain" not in st.session_state:
    chain_with_history, rag_func = create_chain(model_name=selected_model)
    st.session_state["chain"] = chain_with_history
    st.session_state["rag_func"] = rag_func


# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state["chain"]
    rag_func = st.session_state.get("rag_func")

    if chain is not None and rag_func is not None:
        # RAG 함수로 컨텍스트 추가
        inputs = rag_func({"question": user_input})

        response = chain.stream(
            # 질문과 컨텍스트 입력
            inputs,
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        # 사용자의 입력
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        # RAG 시스템 로드 실패 경고 메시지
        warning_msg.error(
            "RAG 시스템을 로드하지 못했습니다. worlds.txt 파일을 확인해주세요."
        )
