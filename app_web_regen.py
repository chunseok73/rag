import os
import shutil
import pandas as pd
import numpy as np
import streamlit as st
# from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# 기준 app_web.py를 확장
# - '캐시 삭제/재생성' 버튼 추가
# - 클릭 시: embedding 디렉토리 삭제 → origin CSV로 em01.csv/em01.npy 재생성
# - 재생성 직후, st.session_state에 (docs_df, doc_emb) 즉시 주입하여
#   streamlit run 재실행 없이 같은 화면에서 바로 '질문하기' 가능
# - UI 진행 표시(사용자 지루함 방지)
# =========================================================

# (0) 경로/환경 설정
APP_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\rag
PROJECTS_DIR = os.path.dirname(APP_DIR)                    # ...\Projects
ENV_PATH = os.path.join(PROJECTS_DIR, ".env")

ORIGIN_DIR = os.path.join(APP_DIR, "origin")
EMB_DIR = os.path.join(APP_DIR, "embedding")

ORIGIN_CSV = os.path.join(ORIGIN_DIR, "company_manual01.csv")  # 원본
CACHE_DOCS = os.path.join(EMB_DIR, "em01.csv")                 # 캐시 복사본
CACHE_EMB = os.path.join(EMB_DIR, "em01.npy")                  # 임베딩 결과

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 Projects 폴더의 .env 파일에 설정되어 있지 않습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)


# (1) 유틸
def ensure_dirs():
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

def cosine_similarity_matrix(emb_matrix: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(q_vec) + 1e-12
    e_norm = np.linalg.norm(emb_matrix, axis=1) + 1e-12
    return (emb_matrix @ q_vec) / (e_norm * q_norm)

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_doc_embeddings_with_ui(df: pd.DataFrame) -> np.ndarray:
    vectors = []
    total = len(df)

    progress = st.progress(0, text="임베딩 준비 중...")
    status = st.empty()

    for i, text in enumerate(df["content"].tolist(), start=1):
        vec = get_embedding(text)  # 쿼터 부족이면 여기서 예외
        vectors.append(vec)

        pct = int(i / total * 100) if total else 100
        progress.progress(pct, text=f"임베딩 생성 중... {i}/{total} ({pct}%)")
        status.write(f"현재 처리 중: {text[:60]}{'...' if len(text) > 60 else ''}")

    progress.progress(100, text="임베딩 생성 완료!")
    status.empty()

    return np.vstack(vectors).astype(np.float32)

def load_index_from_cache() -> tuple[pd.DataFrame, np.ndarray]:
    docs_df = pd.read_csv(CACHE_DOCS)
    emb = np.load(CACHE_EMB)
    if "content" not in docs_df.columns:
        raise ValueError("캐시된 em01.csv에 'content' 컬럼이 없습니다.")
    if emb.shape[0] != len(docs_df):
        raise ValueError("em01.npy 행 수와 em01.csv 문서 수가 일치하지 않습니다.")
    return docs_df, emb

def load_or_create_index_no_ui() -> tuple[pd.DataFrame, np.ndarray]:
    """
    - 기본 로딩용 (기존 app_web.py와 동일 동작)
    - 캐시 있으면 로드
    - 없으면 원본으로부터 생성(이때는 UI 진행 표시 없이 print만)
    """
    ensure_dirs()

    if os.path.exists(CACHE_DOCS) and os.path.exists(CACHE_EMB):
        return load_index_from_cache()

    if not os.path.exists(ORIGIN_CSV):
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {ORIGIN_CSV}")

    df = pd.read_csv(ORIGIN_CSV)
    if "content" not in df.columns:
        raise ValueError("company_manual01.csv에는 최소 'content' 컬럼이 필요합니다.")
    df["content"] = df["content"].astype(str)

    vectors = []
    total = len(df)
    for i, text in enumerate(df["content"].tolist(), start=1):
        vec = get_embedding(text)
        vectors.append(vec)
        print(f"[embedding] {i}/{total} done")

    emb = np.vstack(vectors).astype(np.float32)
    df.to_csv(CACHE_DOCS, index=False)
    np.save(CACHE_EMB, emb)

    return df, emb

def regenerate_cache_from_origin_solution_A() -> tuple[pd.DataFrame, np.ndarray]:
    """
    <조건 + 해결책 A>
    - 먼저 embedding 디렉토리 삭제
    - origin CSV로 embedding 디렉토리 + em01.csv + em01.npy 생성
    - 임베딩 단계 UI 표시
    - 재생성 직후, session_state에 결과를 넣어 즉시 반영 (rerun 최소화/불필요)
    """
    # 0) 원본 확인
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    if not os.path.exists(ORIGIN_CSV):
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {ORIGIN_CSV}")

    # 1) 기존 embedding 디렉토리 삭제
    if os.path.exists(EMB_DIR):
        shutil.rmtree(EMB_DIR)

    # 2) embedding 디렉토리 재생성
    os.makedirs(EMB_DIR, exist_ok=True)

    # 3) 원본 로드
    df = pd.read_csv(ORIGIN_CSV)
    if "content" not in df.columns:
        raise ValueError("company_manual01.csv에는 최소 'content' 컬럼이 필요합니다.")
    df["content"] = df["content"].astype(str)

    # 4) 임베딩 생성(UI 표시)
    emb = build_doc_embeddings_with_ui(df)

    # 5) 캐시 저장
    df.to_csv(CACHE_DOCS, index=False)
    np.save(CACHE_EMB, emb)

    return df, emb

def search_manual(docs_df: pd.DataFrame, doc_emb: np.ndarray, question: str, top_k: int = 3) -> pd.DataFrame:
    q_vec = get_embedding(question)
    sims = cosine_similarity_matrix(doc_emb, q_vec)
    result = docs_df.copy()
    result["similarity"] = sims
    return result.sort_values("similarity", ascending=False).head(top_k)

def ask_llm(question: str, snippets: list[str]) -> str:
    context = "\n".join([f"- {s}" for s in snippets])
    prompt = f"""
당신은 회사 규정을 잘 아는 인사/총무 담당자입니다.
아래 [회사 규정] 근거만 사용해 간결하고 명확하게 답변하세요.
근거에 없는 내용이면 "규정에 명시되어 있지 않습니다."라고 답하세요.

[회사 규정]
{context}

[질문]
{question}

[답변 형식]
- 결론: (한 문장)
- 근거: (규정 문장 인용/요약 1~2개)
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# (2) Streamlit UI
st.set_page_config(page_title="RAG 매뉴얼 Q&A (Cache Regenerate)", layout="wide")
st.title("사내 매뉴얼 Q&A ('캐시 삭제/재생성' 확장)")
st.caption("버튼 클릭 시.. embedding 디렉토리를 삭제하고, company_manual01.csv 기반으로 em01.csv와 em01.npy를 재생성")

# session_state 초기화
if "docs_df" not in st.session_state:
    st.session_state["docs_df"] = None
if "doc_emb" not in st.session_state:
    st.session_state["doc_emb"] = None
if "ready" not in st.session_state:
    st.session_state["ready"] = False
if "is_building" not in st.session_state:
    st.session_state["is_building"] = False

with st.sidebar:
    st.subheader("설정")
    top_k = st.slider("검색 Top-K", 1, 5, 3)
    show_similarity = st.toggle("유사도 표시", value=True)

    st.divider()
    st.subheader("캐시 관리")

    # Solution A: 재생성 후 st.session_state에 즉시 반영
    if st.button("캐시 삭제/재생성", type="primary", disabled=st.session_state["is_building"]):
        try:
            st.session_state["is_building"] = True

            with st.spinner("캐시를 삭제하고 재생성하는 중입니다..."):
                df_new, emb_new = regenerate_cache_from_origin_solution_A()

            # 즉시 반영: rerun 없이 현재 실행 컨텍스트에서 상태 업데이트
            st.session_state["docs_df"] = df_new
            st.session_state["doc_emb"] = emb_new
            st.session_state["ready"] = True

            # Streamlit 캐시 무효화(다음 rerun에서도 새 캐시 파일을 쓰도록)
            st.cache_data.clear()
            st.cache_resource.clear()

            st.success("캐시 삭제/재생성 완료! (즉시 반영됨)")
        except Exception as e:
            st.error(f"재생성 실패: {type(e).__name__}: {e}")
            st.info("429 (insufficient_quota)이면 OpenAI 결제/쿼터 문제입니다.")
        finally:
            st.session_state["is_building"] = False

    # st.divider()
    # st.subheader("파일 경로")
    # st.code(
    #     f".env: {ENV_PATH}\n"
    #     f"origin: {ORIGIN_CSV}\n"
    #     f"cache: {CACHE_DOCS}\n"
    #     f"emb: {CACHE_EMB}"
    # )
    st.divider()
    st.subheader("추천 질문")
    st.code(
        "국내 출장 식비 기준 알려줘\n"
        "연차는 반차로 쓸 수 있어?\n"
        "야근하면 보상은 어떻게 돼?\n"
        "회사 문서를 개인 메일로 보내도 돼?\n"
        "명예퇴직시 퇴직금 정산 방법은?"
    )

# 1) 아직 session_state에 인덱스가 없으면(첫 로드), 파일 캐시에서 준비
if not st.session_state["ready"]:
    try:
        with st.spinner("문서 인덱스를 준비 중입니다... (캐시가 없으면 최초 1회 임베딩 생성)"):
            df_init, emb_init = load_or_create_index_no_ui()
        st.session_state["docs_df"] = df_init
        st.session_state["doc_emb"] = emb_init
        st.session_state["ready"] = True
        st.success(f"인덱스 준비 완료! (문장 수: {len(df_init)})")
    except Exception as e:
        st.error(f"인덱스 준비 실패: {type(e).__name__}: {e}")
        st.stop()
else:
    df_init = st.session_state["docs_df"]
    st.success(f"인덱스 준비 완료! (문장 수: {len(df_init)})")

docs_df = st.session_state["docs_df"]
doc_emb = st.session_state["doc_emb"]


# 2) 질문 UI
question = st.text_input("질문을 입력하세요", placeholder="예: 국내 출장 식비 기준 알려줘")
colA, colB = st.columns([1, 1])

ask = st.button(
    "질문하기",
    type="primary",
    disabled=(not question.strip()) or st.session_state["is_building"] or (docs_df is None) or (doc_emb is None)
)

if ask:
    try:
        with st.spinner("관련 규정 검색 중..."):
            hits = search_manual(docs_df, doc_emb, question, top_k=top_k)
            snippets = hits["content"].astype(str).tolist()

        with st.spinner("답변 생성 중..."):
            answer = ask_llm(question, snippets)

        with colA:
            st.subheader("답변")
            st.write(answer)

        with colB:
            st.subheader("참조 규정")
            view_cols = ["content"]
            if "category" in hits.columns:
                view_cols = ["category", "content"]
            if show_similarity:
                view_cols.append("similarity")

            view = hits[view_cols].copy()
            if "similarity" in view.columns:
                view["similarity"] = view["similarity"].astype(float).round(3)

            st.dataframe(view, use_container_width=True)

    except Exception as e:
        st.error(f"처리 실패: {type(e).__name__}: {e}")
        st.info("429 (insufficient_quota)이면 OpenAI 결제/쿼터 문제입니다.")
