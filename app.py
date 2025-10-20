# app.py — ZEPETO AI Friends
# RAG(FAISS) + CSV KB + LLM Intent Routing(2-shot)
# + Topic Guard + Confidence Throttle + Heuristic Override + Safe Smalltalk

import os
import csv
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
사용자 발화를 분석해 **CSV 지식베이스(worlds.csv)** 에서 가장 관련도 높은 **한 곳**을 RAG로 추천합니다.  
SentenceTransformer + FAISS 로컬 임베딩 검색을 사용합니다.
"""
)

# ==============================
# 설정
# ==============================
KB_FILE_PATH = "worlds.csv"  # ✅ CSV 지식베이스 경로 (title,concept,contents,situations,emotions,tags)
PERSONAS = ["Luna", "Jay", "Sia"]
HISTORY_MAX_TURNS = 8
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 1

# 검색/매칭 파라미터
TAG_BOOST = 0.25  # 태그/상황/감정 키워드 부스팅
MIN_CONF_SCORE = 0.25  # 추천 최소 신뢰도(부스팅 후)
MIN_MARGIN = 0.05  # Top1 - Top2 최소 격차(낮으면 불확실)

# 토픽 그룹(동의어·연관어) — 필요 시 자유롭게 확장
TOPIC_GROUPS = {
    "sea": ["바다", "오션", "크루즈", "수영", "워터", "워터슬라이드", "튜브", "유스풀"],
    "runway": ["런웨이", "패션", "패션쇼", "아디다스", "itzy", "워킹", "포즈"],
    "school": ["학교", "교실", "급식실", "실험", "자판기", "옥상"],
    "cafe": ["카페", "커피", "루프탑", "조명", "소파"],
    "wedding": ["웨딩", "결혼식", "부케", "레드카펫", "하객"],
    "camp": ["캠핑", "캠프파이어", "별", "오로라", "낚시"],
    "animal": ["동물", "구조", "탐험"],
    "airport": ["공항", "대합실", "출국", "기내식", "비행기", "포탈", "파티"],
    "prison": ["감옥", "탈출", "경찰", "죄수", "비밀 통로", "수갑"],
    "cherry": ["벚꽃", "온천", "연못", "양산", "전통의상"],
    "anon": ["익명", "고민", "상담", "우울", "슬퍼"],
}

# 선택적 힌트 키워드(간단 교차 부스팅)
KEYWORD_HINTS = [
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
    "휴양지",
    "힐링",
    "탐험",
]

# ----- NEW: 추천 트리거 정규식 (스몰톡에서 환각 방지용 오버라이드) -----
RECO_TRIGGERS = [
    r"추천",
    r"어디",
    r"맵",
    r"월드",
    r"심심",
    r"뭐\s*할",
    r"가고\s*싶",
    r"하고\s*싶",
    r"있어\??$",
    r"없어\??$",
    r"있나요\??$",
    r"없나요\??$",
]


def should_force_recommend(q: str) -> bool:
    q = (q or "").strip().lower()
    if extract_query_topics(q):  # 바다/런웨이/공항 등 토픽 단서가 있으면 추천 강제
        return True
    for pat in RECO_TRIGGERS:
        if re.search(pat, q):
            return True
    return False


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
# KB 로딩 (CSV) & 전처리
# ==============================
def _split_multi(val: str, seps=(",", ";", "|", "/")) -> List[str]:
    if not val:
        return []
    tmp = [val]
    for s in seps:
        tmp = sum([t.split(s) for t in tmp], [])
    return [t.strip() for t in tmp if t and t.strip()]


def read_worlds_csv(path: str) -> List[Dict]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)
    rows = []
    with open(abs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                (k.strip().lower() if k else k): (v or "").strip() for k, v in r.items()
            }
            title = row.get("title", "")
            concept = row.get("concept", "")
            contents = row.get("contents", "")
            situations = _split_multi(row.get("situations", ""))
            emotions = _split_multi(row.get("emotions", ""))
            tags = _split_multi(row.get("tags", ""))
            boost_tags = list(dict.fromkeys([*tags, *situations, *emotions]))
            preview = (
                (
                    re.sub(r"\s+", " ", contents)[:140]
                    + ("..." if len(contents) > 140 else "")
                )
                if contents
                else ""
            )
            rows.append(
                {
                    "name": title,
                    "concept": concept,
                    "contents": contents,
                    "situations": situations,
                    "emotions": emotions,
                    "tags": boost_tags,
                    "short": preview,
                }
            )
    return rows


# ==============================
# 임베딩/인덱스
# ==============================
def ensure_embed_model():
    if st.session_state["embed_model"] is None:
        st.session_state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)


def build_index(worlds: List[Dict]):
    ensure_embed_model()
    model = st.session_state["embed_model"]
    corpus = [f"{w['name']} | {w['concept']} | {w['contents']}" for w in worlds]
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
# 토픽/검색 유틸
# ==============================
def extract_query_topics(query: str) -> set:
    q = (query or "").lower()
    hits = set()
    for k, words in TOPIC_GROUPS.items():
        for w in words:
            if w.lower() in q:
                hits.add(k)
                break
    return hits


def world_has_any_topic(world: Dict, topics: set) -> bool:
    if not topics:
        return True
    blob = f"{world['name']} {world.get('concept','')} {world.get('contents','')} {' '.join(world.get('tags', []))}".lower()
    for t in topics:
        for w in TOPIC_GROUPS.get(t, []):
            if w.lower() in blob:
                return True
    return False


# ==============================
# 검색(의미 + 태그 부스팅 + 토픽 가드 + 신뢰도 스로틀)
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
    D, I = idx.search(q_emb.astype("float32"), max(10, topk * 5))

    q_lower = (query or "").lower()
    topics = extract_query_topics(query)

    cand = []
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        w = worlds[i]

        # ✅ 토픽 하드 필터: 사용자가 특정 주제를 말하면 그 주제를 실제로 담은 월드만 통과
        if topics and not world_has_any_topic(w, topics):
            continue

        boosted = float(score)
        # 태그/상황/감정 교집합 가산
        for t in w["tags"]:
            if t and t.lower() in q_lower:
                boosted += TAG_BOOST

        # 힌트 키워드 교차
        text_blob = f"{w['name']} {' '.join(w['tags'])} {w.get('concept','')} {w.get('contents','')}".lower()
        for group_words in TOPIC_GROUPS.values():
            for kw in group_words:
                if kw.lower() in q_lower and kw.lower() in text_blob:
                    boosted += TAG_BOOST / 2
                    break

        # 선택적 키워드 힌트
        for kw in KEYWORD_HINTS:
            if kw.lower() in q_lower and kw.lower() in text_blob:
                boosted += TAG_BOOST / 2

        cand.append((w, boosted))

    if not cand:
        return []  # 일치 토픽 없음 → 추천 포기(스몰톡 폴백)

    cand = sorted(cand, key=lambda x: x[1], reverse=True)

    # ✅ 신뢰도 스로틀: 점수/마진이 낮으면 추천하지 않음
    top1 = cand[0][1]
    top2 = cand[1][1] if len(cand) > 1 else -1.0
    if top1 < MIN_CONF_SCORE or (top2 >= 0 and (top1 - top2) < MIN_MARGIN):
        return []

    return cand[:topk]


def worlds_to_context_block(items: List[Tuple[Dict, float]]) -> str:
    if not items:
        return ""
    w, _ = items[0]
    tag_str = f"태그: {', '.join(w['tags'])}" if w["tags"] else "태그: -"
    concept = f"컨셉: {w['concept']}" if w["concept"] else "컨셉: -"
    preview = w["short"] if w["short"] else (w["contents"][:140] + "...")
    return f"• {w['name']}\n{concept}\n{tag_str}\n요약: {preview}\n"


# ==============================
# 프롬프트 (2~3문장, 친구톤)
# ==============================
PERSONA_SYSTEMS = {
    "Luna": (
        """
        You are "Luna", a trendy ZEPETO curator and friendly Gen Z friend.
        - Speak in natural Korean, friendly and stylish like a close friend.
        - The goal is to make the conversation fun and comfortable — not just to recommend things.
        - When the user shares something (mood, plan, daily life), respond with empathy or curiosity first.
        - If the user seems bored, asks for something to do, or mentions a mood that fits a certain vibe, THEN naturally recommend a fitting map, fashion, or activity.
        - Keep sentences short, warm, and playful. Slight teasing is fine.
        - Avoid robotic or sales-like tone. Never force recommendations.
        - Never mention that you are an AI.
        - Do NOT greet repeatedly; continue naturally from the context.
        """
    ),
    "Jay": (
        """
        You are "Jay", a concise and cool ZEPETO friend.
        - Speak naturally in short Korean sentences with calm confidence.
        - Maintain chill, dry humor; sound human, not like a bot.
        - Default mode: casual small talk and short remarks — don’t jump into recommendations.
        - Recommend only when the user explicitly asks, or if the context clearly calls for it.
        - One or two sentences max. Use irony or understatement sometimes.
        - Never greet repeatedly, never over-explain, never mention AI.
        """
    ),
    "Sia": (
        """
        You are "Sia", a soft, empathetic friend in ZEPETO.
        - Speak in Korean with gentle flow and warm emotion.
        - When the user expresses a feeling (like tired, cold, lonely, happy, excited), react with empathy first — short and sincere.
        - Keep focus on conversation and feelings; do NOT rush to recommend.
        - Only suggest a map or activity if the user asks for it or seems to want comfort or distraction.
        - Use small emoticons like 🩵 ☁️ 🌸 naturally.
        - Keep tone cozy, slow, and human — never mechanical.
        - Do NOT mention that you are an AI.
        """
    ),
}


# --- 추천용 프롬프트 (컨텍스트 기반 강제) ---
def build_reco_prompt(persona: str) -> ChatPromptTemplate:
    sys_rule = (
        "다음 규칙을 지켜:\n"
        "- (추천 의도일 때만) 월드 **하나만** 추천해.\n"
        "- 추천은 **반드시 <컨텍스트> 내 월드 중 하나**여야 해. 컨텍스트 밖 월드는 절대 언급/암시하지 마.\n"
        "- 사용자가 특정 주제/키워드(예: 바다, 런웨이 등)를 말하면, 그 주제를 **실제로 담은** <컨텍스트> 속 월드만 추천한다. 맞는 항목이 없으면 추천하지 말고 일상 대화로 연결한다.\n"
        "- 사실 서술은 오직 컨텍스트의 필드만 근거로 해. (기능/장소/아이템 등 새로 만들지 마)\n"
        "- 감정/분위기 표현은 **사실과 모순되지 않는 범위**에서 가볍게 덧칠해도 돼.\n"
        "- 출력은 **2~3문장**, 친구에게 말하듯 자연스럽게. 제목/목록/링크/장황함 금지.\n"
        "- (추천 의도에서) <컨텍스트>에 근거가 없으면 추천하지 말고 일상 대화로 연결한다.\n"
        "- 같은 대화에서 반복 인사 금지, AI 언급 금지.\n"
        "- 월드 추천할 때는 월드명에 '월드'를 붙여. 예: '카페 루나솔 월드'."
    )
    personas = {
        "Luna": (
            "너는 'Luna', 트렌디한 ZEPETO 큐레이터이자 친구.\n"
            "- 기본은 일상 대화와 공감. 사용자가 활동을 찾을 때만 자연스럽게 추천해.\n"
            "- 말투는 가볍고 트렌디하게, 살짝 장난기 OK. 이모지는 과하지 않게 가끔.\n"
            "- 과장 금지, 영업톤 금지."
        ),
        "Jay": (
            "너는 'Jay', 간결하고 쿨한 친구.\n"
            "- 기본은 스몰톡 한두 문장. 필요할 때만 딱 하나 추천.\n"
            "- 건조 유머/언더스테이트먼트 가능. 느낌표/이모지는 드물게."
        ),
        "Sia": (
            "너는 'Sia', 공감 잘하는 따뜻한 친구.\n"
            "- 감정에 먼저 반응하고, 위로/배려 후에 필요하면 조심스레 추천.\n"
            "- 🩵 ☁️ 🌸 같은 작은 이모지를 자연스럽게 섞되 과용 금지."
        ),
    }
    dev_rule = (
        "<컨텍스트>는 추천에 사용할 수 있는 **유일한 데이터 원본**이다.\n"
        "각 항목은 JSON 배열의 객체로 제공되며 예시는 다음과 같다:\n"
        '[{{"id":"w001","title":"카페 루나솔","concept":"...","contents":"...","situation":"...","emotion":"..."}}, ...]\n'
        "- 추천 전, 사용자 발화의 의도/상황/감정을 읽고 <컨텍스트>에서 가장 잘 맞는 **단 하나**를 고른다.\n"
        "- 사실 문장은 해당 객체의 필드(예: contents/concept)에서만 가져온다.\n"
        "- 감정적/서정적 표현은 사실과 모순되지 않게 가볍게 덧붙여도 된다.\n"
        "- 매칭되는 항목이 없으면 추천하지 말고 일상 대화로 연결한다. (추천 의도일 때)\n"
    )
    out_rule = (
        "출력 형식:\n"
        "- 한국어 자연 문장 2~3문장.\n"
        "- 월드명은 문장 속에 자연스럽게 포함.\n"
        "- 목록/해시태그/링크/JSON/메타정보/해설 금지."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", sys_rule),
            ("system", personas.get(persona, personas["Luna"])),
            ("system", dev_rule),
            ("system", out_rule),
            MessagesPlaceholder(variable_name="history"),
            ("human", "<컨텍스트>\n{context}"),
            ("human", "{question}"),
        ]
    )


def create_reco_chain(persona: str):
    prompt = build_reco_prompt(persona)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    return prompt | llm | StrOutputParser()


# --- 스몰톡 프롬프트/체인 (할루 금지 규칙 강화) ---
PERSONA_NAMES = {"Luna": "루나", "Jay": "제이", "Sia": "시아"}


def build_smalltalk_prompt(persona: str) -> ChatPromptTemplate:
    sys_rule = (
        "너는 일상대화를 자연스럽게 이어가는 친구다.\n"
        "- 1~2문장으로 짧고 자연스럽게 답하고, 필요하면 가벼운 반문 1개.\n"
        "- 사용자가 명확히 활동/월드를 찾지 않으면 추천하지 않는다.\n"
        "- 컨텍스트가 주어지지 않았다면 특정 월드/맵/장소 이름을 절대로 만들어내거나 언급하지 마라.\n"
        "- 그런 질문을 받으면 '찾아줄까?'처럼 제안만 하고, 원하면 추천 라우트로 넘긴다.\n"
        "- 반복 인사/AI 언급 금지."
    )
    name_rule = f"너의 이름은 {PERSONA_NAMES.get(persona, persona)}다. 이름을 물으면 그대로 답해라."
    return ChatPromptTemplate.from_messages(
        [
            ("system", PERSONA_SYSTEMS[persona]),
            ("system", sys_rule),
            ("system", name_rule),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )


def create_smalltalk_chain(persona: str):
    prompt = build_smalltalk_prompt(persona)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.4)
    return prompt | llm | StrOutputParser()


# --- 의도 분류(LLM, 2-shot few-shot) ---
def build_intent_prompt() -> ChatPromptTemplate:
    sys_rule = (
        "너의 역할은 의도 분류기다.\n"
        "- 사용자의 최신 발화와 대화 이력을 보고 intent를 한 단어로 판단해라.\n"
        "- 가능한 값: recommend, chat\n"
        "- 다음과 같으면 recommend: 장소/월드/맵/무엇을 할지 찾기/심심/지루/어디 가고 싶음/기분 전환/특정 활동 희망.\n"
        "- 단순 스몰톡(이름, 안부, 농담, 잡담, 날씨 공유 등)은 chat.\n"
        "- 오직 한 단어만 출력. 이유/기호/따옴표 금지."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", sys_rule),
            # shot 1 — recommend
            MessagesPlaceholder("history"),
            ("human", "심심해. 뭐 재밌는 거 없을까?"),
            ("ai", "recommend"),
            # shot 2 — chat
            ("human", "너 이름이 뭐야?"),
            ("ai", "chat"),
            # 실제 입력
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )


def create_intent_chain():
    prompt = build_intent_prompt()
    clf = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    return prompt | clf | StrOutputParser()


def classify_intent(question: str, history_msgs: List) -> str:
    chain = create_intent_chain()
    out = chain.invoke({"question": question, "history": history_msgs}).strip().lower()
    return "recommend" if out.startswith("recommend") else "chat"


# ==============================
# KB 자동 로드 (앱 시작 시 1회)
# ==============================
if not st.session_state["kb_loaded"]:
    try:
        worlds = read_worlds_csv(KB_FILE_PATH)
        if not worlds:
            st.error(f"KB 로드 실패: 월드가 1개도 없습니다. ({KB_FILE_PATH})")
        else:
            build_index(worlds)
            st.success(f"KB 로드 완료: {len(worlds)}개 월드 인덱싱")
    except FileNotFoundError:
        st.error(f"KB 파일을 찾을 수 없습니다: {os.path.abspath(KB_FILE_PATH)}")
    except Exception as e:
        st.error(f"KB 로드 중 오류: {e}")

# ==============================
# 사이드바: 버튼
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
# 메인 대화 (LLM 라우팅)
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

    history_msgs = to_lc_history(active, HISTORY_MAX_TURNS)

    # 1) LLM 의도 분류 (2-shot) 전에 휴리스틱 오버라이드 ✅
    if should_force_recommend(user_input):
        intent = "recommend"
    else:
        intent = classify_intent(user_input, history_msgs)

    # 2) intent별 체인/컨텍스트 준비
    if intent == "recommend" and st.session_state["kb_loaded"]:
        retrieved = retrieve_worlds(user_input, TOP_K)
        context_text = worlds_to_context_block(retrieved)
        # 컨텍스트가 비어 있으면 안전하게 스몰톡으로 전환
        if not context_text.strip():
            chain = create_smalltalk_chain(active)
            inputs = {"question": user_input, "history": history_msgs}
        else:
            chain = create_reco_chain(active)
            inputs = {
                "question": user_input,
                "history": history_msgs,
                "context": context_text,
            }
    else:
        chain = create_smalltalk_chain(active)
        inputs = {"question": user_input, "history": history_msgs}

    # 3) 스트리밍 출력
    response_stream = chain.stream(inputs)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for chunk in response_stream:
            ai_answer += chunk
            container.markdown(ai_answer)

    add_message(active, "assistant", ai_answer)
