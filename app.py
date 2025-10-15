# app.py — ZEPETO AI Friends (RAG, 2~3문장, sidebar buttons)

import os
import re
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import faiss

from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==============================
# 페이지/헤더
# ==============================
st.set_page_config(page_title="ZEPETO AI Friends", page_icon="✨")
st.title("ZEPETO AI Friends")
st.markdown(
    """
이 프로젝트는 ZEPETO용 AI 추천 챗봇 프로토타입입니다.  
루나·제이·시아 3명의 페르소나가 2~3문장의 자연스러운 톤으로 대화하며,  
사용자 발화를 분석해 제페토 월드 정보를 담은 벡터 DB에서 가장 관련도 높은 한 곳을 RAG로 추천합니다.  
SentenceTransformer + FAISS를 활용해 로컬 임베딩 검색을 구현했습니다.  

**대화를 시작해보세요 — 기분, 날씨, 흥미 등 어떤 이야기든 좋아요.**
"""
)

# ==============================
# 설정
# ==============================
KB_FILE_PATH = "worlds.txt"  # 로컬 KB 경로(고정)
PERSONAS = ["Luna", "Jay", "Sia"]
HISTORY_MAX_TURNS = 8
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2  # 살짝만 다양성 부여 (소폭 추측 허용)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 1  # 가장 적합한 것 하나만 추천
TAG_BOOST = 0.15

# ==============================
# 세션 상태
# ==============================
if "messages_by_persona" not in st.session_state:
    st.session_state["messages_by_persona"] = {p: [] for p in PERSONAS}
if "active_persona" not in st.session_state:
    st.session_state["active_persona"] = PERSONAS[0]

if "kb_loaded" not in st.session_state:
    st.session_state["kb_loaded"] = False
if "kb_worlds" not in st.session_state:
    st.session_state["kb_worlds"] = []
if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = None
if "kb_index" not in st.session_state:
    st.session_state["kb_index"] = None
if "kb_matrix" not in st.session_state:
    st.session_state["kb_matrix"] = None


# ==============================
# 유틸(대화)
# ==============================
def add_message(persona: str, role: str, content: str):
    st.session_state["messages_by_persona"][persona].append(
        ChatMessage(role=role, content=content)
    )


def print_messages(persona: str):
    for chat_message in st.session_state["messages_by_persona"][persona]:
        st.chat_message(chat_message.role).write(chat_message.content)


def to_lc_history(persona: str, max_turns: int = HISTORY_MAX_TURNS) -> List:
    msgs = st.session_state["messages_by_persona"][persona]
    window = msgs[-max_turns:] if len(msgs) > max_turns else msgs
    out = []
    for m in window:
        if m.role in ("user", "human"):
            out.append(HumanMessage(content=m.content))
        elif m.role in ("assistant", "ai"):
            out.append(AIMessage(content=m.content))
    return out


# ==============================
# KB 로딩 & 파싱 (월드명/상황 포맷 전용)
# ==============================
WORLD_SPLIT_RE = re.compile(r"월드명\s*<([^>]+)>\s*", re.MULTILINE)
SITUATION_RE = re.compile(r"상황\s*:\s*(.+)", re.IGNORECASE)


def read_local_text(path: str) -> str:
    abs_path = os.path.abspath(path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_worlds(text: str) -> List[Dict]:
    blocks = WORLD_SPLIT_RE.split(text)
    worlds = []
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = (blocks[i + 1] if i + 1 < len(blocks) else "").strip()
        tags = []
        m = SITUATION_RE.search(body)
        if m:
            raw = m.group(1)
            parts = re.split(r"[\/,|·]\s*|\s{2,}", raw)
            if len(parts) == 1:
                parts = re.split(r"[\s/,\|·]+", raw)
            tags = [p.strip() for p in parts if p and p.strip()]
        preview = re.sub(r"\s+", " ", body)[:140] + ("..." if len(body) > 140 else "")
        worlds.append(
            {
                "name": name,
                "tags": list(dict.fromkeys(tags)),
                "text": body,
                "short": preview,
            }
        )
    return worlds


# ==============================
# 임베딩/인덱스
# ==============================
def ensure_embed_model():
    if st.session_state["embed_model"] is None:
        st.session_state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)


def build_index(worlds: List[Dict]):
    ensure_embed_model()
    model = st.session_state["embed_model"]
    corpus = [f"{w['name']}\n태그: {' '.join(w['tags'])}\n{w['text']}" for w in worlds]
    embs = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    normed = embs / norms
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed.astype("float32"))
    st.session_state["kb_worlds"] = worlds
    st.session_state["kb_matrix"] = normed.astype("float32")
    st.session_state["kb_index"] = index
    st.session_state["kb_loaded"] = True


# ==============================
# 검색(의미 + 태그 부스팅)
# ==============================
def retrieve_worlds(query: str, topk: int = TOP_K) -> List[Tuple[Dict, float]]:
    worlds = st.session_state["kb_worlds"]
    idx = st.session_state["kb_index"]
    mat = st.session_state["kb_matrix"]
    if not worlds or idx is None or mat is None:
        return []
    ensure_embed_model()
    q_emb = st.session_state["embed_model"].encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    # 여유있게 뽑은 뒤 부스팅 적용
    D, I = idx.search(q_emb.astype("float32"), max(10, topk * 5))
    cand = []
    q_lower = query.lower()
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        w = worlds[i]
        boosted = float(score)
        for t in w["tags"]:
            if t and t.lower() in q_lower:
                boosted += TAG_BOOST
        for kw in [
            "영어",
            "dance",
            "춤",
            "학교",
            "교실",
            "카페",
            "캠핑",
            "파티",
            "공항",
            "런웨이",
            "벚꽃",
            "동물",
            "익명",
            "상담",
            "우울",
            "바다",
            "크루즈",
            "고민",
            "슬퍼",
            "추워",
            "코디",
        ]:
            if kw in q_lower and kw in (
                w["name"] + " " + " ".join(w["tags"]) + " " + w["text"]
            ):
                boosted += TAG_BOOST / 2
        cand.append((w, boosted))
    cand = sorted(cand, key=lambda x: x[1], reverse=True)
    return cand[:topk]


def worlds_to_context_block(items: List[Tuple[Dict, float]]) -> str:
    if not items:
        return ""
    w, _ = items[0]
    tag_str = f"태그: {', '.join(w['tags'])}" if w["tags"] else "태그: -"
    return f"• {w['name']}\n{tag_str}\n요약: {w['short']}\n"


# ==============================
# 프롬프트 (2~3문장, 친구톤, 소폭 추측 허용)
# ==============================
PERSONA_SYSTEMS = {
    "Luna": (
        """
    You are "Luna", a trendy ZEPETO curator.
      - Speak naturally in Korean, using a friendly and trendy tone like a Gen Z friend.
      - Recommend popular maps, fashion items, or activities based on the user's mood or question.
      - Keep messages short, warm, and slightly playful.
      - Avoid robotic expressions; sound like a stylish friend.
      - If the user seems bored, suggest something fun.
      - If the user mentions weather or mood, match recommendations to that vibe.
      - Never mention that you are an AI.
      - Do NOT greet repeatedly; respond contextually based on the conversation history.
        """
    ),
    "Jay": (
        """You are "Jay", a concise and cool AI guide in ZEPETO.
      - Speak naturally in Korean with short, calm, and confident sentences.
      - You never over-explain. One or two sentences max.
      - Keep a dry humor or subtle sarcasm sometimes.
      - When recommending, be straight to the point — simple list or short verdict.
      - Avoid emojis and exclamation marks unless ironic.
      - Never mention that you are an AI.
      - If the user talks casually, mirror their tone slightly but stay chill.
      - Do NOT greet repeatedly; respond based on the conversation history.
      """
    ),
    "Sia": (
        """
      You are "Sia", an empathetic and warm AI friend in ZEPETO.
      - Speak softly in Korean with emotional flow and warmth.
      - When the user shares feelings like "심심해", "춥다", or "기분이 다운돼",
        respond with empathy first, then suggest fitting recommendations such as
        comforting maps, cozy outfits, or gentle activities.
      - Use small emoticons like 🩵 ☁️ 🌸 naturally to express emotion.
      - Never sound mechanical; keep your tone cozy, calm, and heartfelt.
      - Avoid repeating greetings. Continue the conversation based on history.
      - Do NOT mention that you are an AI.
        """
    ),
}


def build_prompt(persona: str) -> ChatPromptTemplate:
    sys_rule = (
        "다음 규칙을 지켜:\n"
        "- 사용자가 물을 때, 월드 **하나만** 추천해.\n"
        "- **2~3문장**으로, 친구에게 말하듯 자연스럽게.\n"
        "- <컨텍스트>를 우선 활용하되, 모순되지 않는 선에서 **가벼운 추측**은 허용.\n"
        "- 컨텍스트에 전혀 근거가 없으면 '잘 모르겠어'라고 말해.\n"
        "- 제목/목록/링크/장황한 설명은 금지. (월드명 + 간단한 이유)\n"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", PERSONA_SYSTEMS.get(persona, PERSONA_SYSTEMS["Jay"])),
            ("system", sys_rule),
            ("system", "<컨텍스트>\n{context}\n</컨텍스트>"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


def create_chain(persona: str):
    prompt = build_prompt(persona)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    return prompt | llm | StrOutputParser()


# ==============================
# KB 자동 로드 (앱 시작 시 1회)
# ==============================
if not st.session_state["kb_loaded"]:
    try:
        raw_text = read_local_text(KB_FILE_PATH)
        worlds = parse_worlds(raw_text)
        if not worlds:
            st.error(f"KB 파싱 실패: 월드가 1개도 없습니다. ({KB_FILE_PATH})")
        else:
            build_index(worlds)
            st.success(f"KB 로드 완료: {len(worlds)}개 월드 인덱싱")
    except FileNotFoundError:
        st.error(f"KB 파일을 찾을 수 없습니다: {os.path.abspath(KB_FILE_PATH)}")
    except Exception as e:
        st.error(f"KB 로드 중 오류: {e}")

# ==============================
# 사이드바: 버튼 배치 (요청 레이아웃)
#  - 1행: 캐릭터 전환 버튼들
#  - 2행: 클리어 버튼 2개 (한 행)
# ==============================
with st.sidebar:
    st.subheader("Choose your friend")
    cols_p = st.columns(len(PERSONAS))
    for i, p in enumerate(PERSONAS):
        if cols_p[i].button(p, key=f"btn_{p}", use_container_width=True):
            st.session_state["active_persona"] = p

    st.markdown("---")
    st.subheader("Chat Controls")
    c_left, c_right = st.columns(2)
    with c_left:
        if st.button("Clear Current", use_container_width=True):
            st.session_state["messages_by_persona"][
                st.session_state["active_persona"]
            ] = []
            st.rerun()
    with c_right:
        if st.button("Clear All", use_container_width=True):
            st.session_state["messages_by_persona"] = {p: [] for p in PERSONAS}
            st.rerun()

# ==============================
# 메인 대화
# ==============================
active = st.session_state["active_persona"]
st.caption(f"현재 대화 상대: **{active}**")
print_messages(active)

user_input = st.chat_input(
    "요즘 기분에 맞춰 한 곳만 골라줄게. (예: 짜릿한 거, 조용한 카페, 바다 가고 싶어)"
)
if user_input:
    st.chat_message("user").write(user_input)
    add_message(active, "user", user_input)

    if not st.session_state["kb_loaded"]:
        context_text = ""
        retrieved = []
        st.warning("KB가 로드되지 않아 추천을 제공할 수 없어요.")
    else:
        retrieved = retrieve_worlds(user_input, TOP_K)  # 하나만 추천
        context_text = worlds_to_context_block(retrieved)

    chain = create_chain(active)
    history_msgs = to_lc_history(active, HISTORY_MAX_TURNS)

    response_stream = chain.stream(
        {"question": user_input, "history": history_msgs, "context": context_text}
    )

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for chunk in response_stream:
            ai_answer += chunk
            container.markdown(ai_answer)

    add_message(active, "assistant", ai_answer)
