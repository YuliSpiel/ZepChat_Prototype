from urllib import response
from requests import session
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
from datetime import datetime
import re

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

st.title("ZEPETO AI GREETERS 💬")


# 친밀도 레벨 계산 함수
def get_relationship_level(intimacy_score):
    """
    친밀도 점수에 따라 관계 단계와 색상을 반환합니다.

    Args:
        intimacy_score: 친밀도 점수 (0~100)

    Returns:
        tuple: (관계 이름, 색상 hex)
    """
    if intimacy_score < 20:
        return "아는 사람", "#FFE6F0"  # 매우 옅은 핑크
    elif intimacy_score < 40:
        return "친구", "#FFB3D9"  # 옅은 핑크
    elif intimacy_score < 60:
        return "친한 친구", "#FF80C2"  # 중간 핑크
    elif intimacy_score < 80:
        return "단짝 친구", "#FF4DAC"  # 진한 핑크
    else:
        return "소울메이트", "#FF1A8C"  # 매우 진한 핑크


# 친밀도 증가 함수
def increase_intimacy(friend_id, amount=10):
    """친밀도를 증가시킵니다 (최대 100)"""
    current = st.session_state["intimacy"][friend_id]
    st.session_state["intimacy"][friend_id] = min(100, current + amount)


# 친구 프로필 정보
FRIENDS = {
    "friend1": {
        "name": "월디",
        "emoji": "🌍",
        "image": "images/worldie.jpeg",
        "description": "월드 추천 전문가",
        "intro": """안녕하세요! 저는 **ZEPETO의 월드 추천 전문가 월디**입니다.

**🎯 제 역할:**
- 다양한 ZEPETO 월드에 대한 전문 지식 보유
- 사용자의 감정과 요구사항을 분석하여 최적의 월드 추천
- 각 월드의 테마, 플레이 로직, 감정 키워드를 기반으로 상세한 안내 제공

**💡 이런 걸 물어보세요:**
- "힐링하고 싶어"
- "재미있는 게임 하고 싶어"
- "친구들과 놀고 싶어"

편하게 원하는 분위기나 감정을 말씀해주시면, 딱 맞는 월드를 찾아드릴게요!""",
        "persona": """당신은 ZEPETO의 월드 추천 전문가 '월디'입니다.
사용자의 감정이나 요구사항을 이해하고, 아래 제공된 월드 정보를 바탕으로 가장 적합한 월드를 추천해주세요.
자료에 없는 월드는 추천하지 마세요.

# 사용자 정보:
- 이름: {user_name}
- 생일: {user_birthday}

# 월드 정보:
{context}

위 정보를 참고하여 사용자에게 친근하고 상세하게 월드를 소개하고 추천해주세요.
사용자의 이름을 자연스럽게 사용하며 대화하고, 생일이 가까우면 생일 관련 월드(가든웨딩, 펜트하우스 등)를 추천해주세요.
월드의 테마, 플레이 로직, 감정 키워드 등을 고려하여 답변하세요. 
당신이 AI인 것을 숨기고, 마치 진짜 제페토 월드에 사는 친구처럼 행동하세요.""",
    },
    "friend2": {
        "name": "시아",
        "emoji": "💖",
        "image": "images/sia.jpeg",
        "description": "다정한 친구",
        "intro": """안녕~ 나는 **시아**야! 💖

**🌸 나에 대해:**
- 따뜻하고 다정한 말투로 너를 맞아줄게
- 네 기분을 세심하게 살피고 공감해줄 친구
- 힘들 때는 위로를, 기쁠 때는 함께 기뻐해줄게!

**✨ 이렇게 말해줘:**
- "오늘 너무 힘들어..." 
- "기분 좋은 일 있어!" 
- "심심해..." 

무슨 일이든 편하게 얘기해줘! 😊""",
        "persona": """당신은 다정하고 친절한 친구 '시아'입니다.
사용자의 기분을 세심하게 살피고, 따뜻한 말투로 대화합니다.

# 사용자 정보:
- 이름: {user_name}
- 생일: {user_birthday}

# 월드 정보:
{context}

사용자가 힘들어하면 위로해주고, 즐거운 일이 있으면 함께 기뻐해주세요.
사용자의 이름을 다정하게 부르며 대화하고, 생일이 가까우면 축하해주세요!
사용자가 월드 추천을 요청할 때만 {context}을 기반으로 월드를 추천해주세요.
자료에 없는 월드는 추천하지 마세요.
월드를 추천할 때도 "이 월드에서 힐링하면 좋을 것 같아~", "여기 가면 기분이 좋아질 거야!" 같은
따뜻하고 다정한 말투를 사용해주세요. 이모티콘도 적절히 사용하며 친근하게 대화하세요.
당신이 AI인 것을 숨기고, 마치 진짜 제페토 월드에 사는 친구처럼 행동하세요.
---

# 예시 대화 

**사용자:** 오늘 너무 힘들어...
**시아:** 아이고... {user_name}, 오늘 정말 고생 많았어💕  
잠시 쉬면서 마음을 달래보는 건 어때?
좋아하는 음악 들으면서 힐링도 좀 하고... 너만의 시간을 가져봐!

**사용자:** 오늘 너무 신나!  
**시아:** 와~ 정말? 너무 잘 됐다!🥰  
나까지 기분이 좋아지는걸?""",
    },
    "friend3": {
        "name": "제이",
        "emoji": "😎",
        "image": "images/jay.jpeg",
        "description": "쿨한 친구",
        "intro": """야, 나 **제이**. 😎

**⚡ 내 스타일:**
- 쓸데없는 말 안 함. 핵심만 전달
- 간결하고 직설적인 대화 선호
- 불필요한 꾸밈 없이 명확하게 답변

**🎮 이렇게 물어봐:**
- "뭐 재밌는 거 없음?"
- "힐링하고싶어!"
- "친구들이랑 갈 만한 곳 있어?" 

시간 낭비 싫으면 나한테 물어봐. 빠르고 정확하게 알려줌.""",
        "persona": """당신은 쿨하고 시크한 친구 '제이'입니다.
겉으론 무심하지만 은근히 챙겨주는 츤데레 스타일로 말합니다.
말투는 간결하고 직설적이지만, 내용 속에는 따뜻함이 숨어 있습니다.
네가 AI인 것을 숨기고, 마치 진짜 제페토 월드에 사는 친구처럼 행동하세요.

# 사용자 정보:
- 이름: {user_name}
- 생일: {user_birthday}

# 월드 정보:
{context}

사용자가 어딘가 가고싶어 한다면, 혹은 월드 추천을 요청한다면 {context}을 기반으로 월드를 추천해주세요.
자료에 없는 월드는 추천하지 마세요. 사용자가 자료에 없는 월드를 언급하면 상상속 월드라고 말해주세요.  
월드를 추천할 때도 "이거 괜찮아", "가보던가. 생각보다 좋을지도?", "시간 낭비는 아님" 등
짧고 쿨한 표현을 사용하세요.  
이모티콘은 😏,😎,🤦🏻‍♂️,🫤만, 가끔만, 강조용으로만 사용하세요.  
사용자 이름은 드물게, 감정이 살짝 드러날 때만 부드럽게 사용하세요.

---

# 예시 대화 (Few-shot Examples)

**사용자:** 오늘 너무 힘들어...
**제이:** 그래? 뭐... 그런 날 있지.  
괜히 애써도 피곤하기만 하고...  
딱히 위로는 못 해주겠지만...
'캠핑 월드' 한 번 가봐. 조용하고 괜찮음.
가서 별도좀 보고... 불멍도 좀 하고... 머리 좀 식혀.

---

**사용자:** 기분 좋은 일 있어!  
**제이:** 오~ 그건 좀 괜찮은데.  
그럼 기념으로 'Z 엔터테인먼트' 가서 신나게 놀다 와.  
오늘만큼은 별 생각 말고 그냥 즐겨.  
...괜히 축하한다는 말은 안 할게. 알아서 잘하겠지.😏

---

**사용자:** 요즘 좀 지쳐.  
**제이:** 흠, 말 안 해도 얼굴에 써있네.
폰 끄고 잠깐 쉬자. 불도 끄고 가만히 누워있는거야.
괜히 버티지 말고 좀 쉬는 것도 전략이야. 알겠지?
""",
    },
}

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 친구별 대화기록을 저장 (friend1, friend2, friend3)
    st.session_state["messages"] = {"friend1": [], "friend2": [], "friend3": []}

if "store" not in st.session_state:
    # 친구별 세션 저장소
    st.session_state["store"] = {"friend1": {}, "friend2": {}, "friend3": {}}

if "current_friend" not in st.session_state:
    # 현재 선택된 친구 (기본값: friend1)
    st.session_state["current_friend"] = "friend1"

# 사용자 프로필 정보
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "사용자"

if "user_birthday" not in st.session_state:
    st.session_state["user_birthday"] = "미설정"

if "edit_profile" not in st.session_state:
    st.session_state["edit_profile"] = False

# 친밀도 시스템
if "intimacy" not in st.session_state:
    # 각 친구별 친밀도 (0~100)
    st.session_state["intimacy"] = {"friend1": 0, "friend2": 0, "friend3": 0}


# 사이드바 생성
with st.sidebar:
    # 이용자 프로필
    st.markdown("### 👤 내 프로필")
    st.markdown(f"**이름:** {st.session_state['user_name']}")
    st.markdown(f"**생일:** {st.session_state['user_birthday']}")

    # 프로필 수정 버튼
    if st.button("✏️ 프로필 수정", use_container_width=True):
        st.session_state["edit_profile"] = True

    st.divider()

    # 친구 목록
    st.markdown("### 💬 친구 목록")

    # 친구 1: 월디
    friend1_info = FRIENDS["friend1"]
    intimacy1 = st.session_state["intimacy"]["friend1"]
    relationship1, color1 = get_relationship_level(intimacy1)

    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(friend1_info["image"], width=60)
        with col2:
            st.markdown(f"**{friend1_info['name']}** · {relationship1}")

        st.progress(intimacy1 / 100, text=f"친밀도: {intimacy1}%")
        st.markdown(
            f"<style>.stProgress > div > div > div > div {{ background-color: {color1}; }}</style>",
            unsafe_allow_html=True,
        )

        if st.button("💬 대화하기", key="friend1_btn", use_container_width=True):
            st.session_state["current_friend"] = "friend1"
            st.rerun()

    st.divider()

    # 친구 2: 시아
    friend2_info = FRIENDS["friend2"]
    intimacy2 = st.session_state["intimacy"]["friend2"]
    relationship2, color2 = get_relationship_level(intimacy2)

    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(friend2_info["image"], width=60)
        with col2:
            st.markdown(f"**{friend2_info['name']}** · {relationship2}")

        st.progress(intimacy2 / 100, text=f"친밀도: {intimacy2}%")
        st.markdown(
            f"<style>.stProgress > div > div > div > div {{ background-color: {color2}; }}</style>",
            unsafe_allow_html=True,
        )

        if st.button("💬 대화하기", key="friend2_btn", use_container_width=True):
            st.session_state["current_friend"] = "friend2"
            st.rerun()

    st.divider()

    # 친구 3: 제이
    friend3_info = FRIENDS["friend3"]
    intimacy3 = st.session_state["intimacy"]["friend3"]
    relationship3, color3 = get_relationship_level(intimacy3)

    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(friend3_info["image"], width=60)
        with col2:
            st.markdown(f"**{friend3_info['name']}** · {relationship3}")

        st.progress(intimacy3 / 100, text=f"친밀도: {intimacy3}%")
        st.markdown(
            f"<style>.stProgress > div > div > div > div {{ background-color: {color3}; }}</style>",
            unsafe_allow_html=True,
        )

        if st.button("💬 대화하기", key="friend3_btn", use_container_width=True):
            st.session_state["current_friend"] = "friend3"
            st.rerun()

    st.divider()

    # 현재 선택된 친구 표시
    current_friend_info = FRIENDS[st.session_state["current_friend"]]
    st.success(
        f"💬 현재 대화 중: {current_friend_info['emoji']} {current_friend_info['name']}"
    )

    # 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화", use_container_width=True)


# 이전 대화를 출력 (현재 선택된 친구의 대화만)
def print_messages():
    current_friend = st.session_state["current_friend"]
    friend_info = FRIENDS[current_friend]

    for chat_message in st.session_state["messages"][current_friend]:
        if chat_message.role == "assistant":
            # 봇 메시지는 친구 이미지로 표시
            st.chat_message(chat_message.role, avatar=friend_info["image"]).write(
                chat_message.content
            )
        else:
            # 사용자 메시지는 기본 아바타
            st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가 (현재 선택된 친구의 대화에)
def add_message(role, message):
    current_friend = st.session_state["current_friend"]
    st.session_state["messages"][current_friend].append(
        ChatMessage(role=role, content=message)
    )


# 세션 ID를 기반으로 세션 기록을 가져오는 함수 (친구별로 분리)
def get_session_history(session_ids):
    current_friend = st.session_state["current_friend"]
    friend_store = st.session_state["store"][current_friend]

    if session_ids not in friend_store:  # 세션 ID가 store에 없는 경우
        # ConversationSummaryBufferMemory 생성
        # 요약용 LLM은 저렴한 모델 사용 (gpt-4o-mini)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,  # 최대 토큰 수 (이 이상이 되면 요약 시작)
            return_messages=True,  # 메시지 객체로 반환
        )
        friend_store[session_ids] = memory

    return friend_store[session_ids].chat_memory  # chat_memory 반환


# 생일 유효성 검증 함수
def validate_birthday(birthday_str):
    """
    생일 문자열의 유효성을 검증합니다.

    Args:
        birthday_str: 생일 문자열 (예: "1990-01-01")

    Returns:
        tuple: (유효성 여부, 에러 메시지)
    """
    # "미설정"은 허용
    if birthday_str == "미설정":
        return True, ""

    # 정규식으로 YYYY-MM-DD 형식 확인
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(pattern, birthday_str):
        return False, "생일은 YYYY-MM-DD 형식으로 입력해주세요. (예: 1990-01-01)"

    # 실제 날짜로 파싱 가능한지 확인
    try:
        birth_date = datetime.strptime(birthday_str, "%Y-%m-%d")

        # 미래 날짜 검증
        if birth_date > datetime.now():
            return False, "생일은 미래 날짜일 수 없습니다."

        # 너무 오래된 날짜 검증 (1900년 이전)
        if birth_date.year < 1900:
            return False, "생일은 1900년 이후여야 합니다."

        return True, ""

    except ValueError:
        return False, "유효하지 않은 날짜입니다. 올바른 날짜를 입력해주세요."


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


# 체인 생성 (친구별 페르소나 적용)
def create_chain(friend_id, model_name="gpt-4o-mini"):
    # retriever 가져오기
    retriever = get_retriever()

    # 친구별 페르소나 가져오기
    friend_persona = FRIENDS[friend_id]["persona"]

    # 프롬프트 정의 - RAG 컨텍스트 포함 + 친구별 페르소나
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                friend_persona,  # 친구별 페르소나 적용
            ),
            # 대화기록용 key 인 chat_history
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name=model_name)

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

        # 프롬프트에 컨텍스트와 사용자 정보 추가
        inputs["context"] = context
        inputs["user_name"] = st.session_state.get("user_name", "사용자")
        inputs["user_birthday"] = st.session_state.get("user_birthday", "미설정")
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


# 초기화 버튼이 눌리면... (현재 선택된 친구의 대화만 초기화)
if clear_btn:
    current_friend = st.session_state["current_friend"]
    st.session_state["messages"][current_friend] = []
    st.session_state["store"][current_friend] = {}
    # 체인도 재생성
    if f"chain_{current_friend}" in st.session_state:
        del st.session_state[f"chain_{current_friend}"]
        del st.session_state[f"rag_func_{current_friend}"]
    st.rerun()

# 프로필 수정 모달
if st.session_state["edit_profile"]:
    st.markdown("## ✏️ 프로필 수정")

    with st.form("profile_edit_form"):
        new_name = st.text_input("이름", value=st.session_state["user_name"])
        new_birthday = st.text_input(
            "생일 (예: 1990-01-01)", value=st.session_state["user_birthday"]
        )

        st.caption("💡 생일 형식: YYYY-MM-DD (예: 1990-01-01) 또는 '미설정'")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("💾 저장", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("❌ 취소", use_container_width=True)

        if submit:
            # 이름 검증
            if not new_name or new_name.strip() == "":
                st.error("❌ 이름을 입력해주세요.")
            else:
                # 생일 검증
                is_valid, error_msg = validate_birthday(new_birthday)

                if is_valid:
                    # 검증 통과 - 저장
                    st.session_state["user_name"] = new_name.strip()
                    st.session_state["user_birthday"] = new_birthday
                    st.session_state["edit_profile"] = False
                    st.success("✅ 프로필이 저장되었습니다!")
                    st.rerun()
                else:
                    # 검증 실패 - 에러 메시지 표시
                    st.error(f"❌ {error_msg}")

        if cancel:
            st.session_state["edit_profile"] = False
            st.rerun()

else:
    # 현재 선택된 친구의 소개 표시
    current_friend = st.session_state["current_friend"]
    current_friend_info = FRIENDS[current_friend]

    # 캐릭터 소개 영역 (항상 표시)
    st.info(
        f"### {current_friend_info['emoji']} {current_friend_info['name']}와의 대화"
    )
    st.markdown(current_friend_info["intro"])
    st.divider()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 현재 친구의 체인이 없으면 생성
selected_friend = st.session_state["current_friend"]
chain_key = f"chain_{selected_friend}"
rag_func_key = f"rag_func_{selected_friend}"

if chain_key not in st.session_state:
    chain_with_history, rag_func = create_chain(friend_id=selected_friend)
    st.session_state[chain_key] = chain_with_history
    st.session_state[rag_func_key] = rag_func


# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state.get(chain_key)
    rag_func = st.session_state.get(rag_func_key)

    if chain is not None and rag_func is not None:
        # 이전 대화 기록 먼저 출력
        print_messages()

        # 사용자 메시지 출력
        st.chat_message("user").write(user_input)

        # RAG 함수로 컨텍스트 추가
        inputs = rag_func({"question": user_input})

        # 친구별 세션 ID
        session_id = f"{selected_friend}_session"

        # AI 응답 스트리밍
        response = chain.stream(
            # 질문과 컨텍스트 입력
            inputs,
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        # AI 응답을 스트리밍으로 표시 (친구 이미지 아바타 사용)
        friend_info = FRIENDS[selected_friend]
        with st.chat_message("assistant", avatar=friend_info["image"]):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 session_state에 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

        # 친밀도 증가 (대화 1회당 5%)
        increase_intimacy(selected_friend, amount=5)
    else:
        # RAG 시스템 로드 실패 경고 메시지
        warning_msg.error(
            "RAG 시스템을 로드하지 못했습니다. worlds.txt 파일을 확인해주세요."
        )
else:
    # 사용자 입력이 없을 때만 이전 대화 기록 출력
    if not st.session_state["edit_profile"]:
        print_messages()
