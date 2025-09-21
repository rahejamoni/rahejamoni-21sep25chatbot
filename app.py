import os
import pickle
import re
import difflib
from datetime import timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# LOAD SECRETS
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EXCEL_QA_PATH = st.secrets["EXCEL_QA_PATH"]
EXCEL_LAN_PATH = st.secrets["EXCEL_LAN_PATH"]
LOG_FILE = st.secrets["LOG_FILE"]

# =========================
# CONFIG
# =========================
EMBED_CACHE = "data_cache/qa_embeddings.pkl"
os.makedirs("data_cache", exist_ok=True)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_WORDS = 150

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# UTILITIES
# =========================
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def _count_words(text: str) -> int:
    return len(text.strip().split())

def _truncate_to_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words])

def _norm(s: str) -> str:
    return str(s).strip().lower()

# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_qa(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    # The merged code has a more robust way to handle column names, so we'll stick to that
    required_cols = {"id", "Business", "Question", "Answer"}
    df = df.rename(columns={
        "questions": "Question",
        "answers": "Answer",
        "Business": "Business"
    })
    # Handle the 'id' column if it's missing by adding it
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel must have columns {required_cols}. Found: {set(df.columns)}")
    return df[["id", "Question", "Answer", "Business"]]


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def build_or_load_embeddings(excel_path=EXCEL_QA_PATH, cache_path=EMBED_CACHE, force_refresh=False):
    df = load_qa(excel_path)
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        if saved.get("csv_len") == len(df):
            return saved["df"], saved["embeddings"]
    corpus = [f"Business: {b}\nQ: {q}\nA: {a}" for q, a, b in zip(df["Question"], df["Answer"], df["Business"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)
    return df, vecs

def retrieve(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = TOP_K):
    q_vec = np.array(embed_texts([query])[0], dtype=np.float32)
    sims = np.array([_cosine_sim(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "id": int(row["id"]),
            "question": str(row["Question"]),
            "answer": str(row["Answer"]),
            "business": str(row["Business"]),
            "score": float(sims[idx])
        })
    return results

# =========================
# LAN STATUS
# =========================
def load_lan_status(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, dtype={"Lan Id": str})
    df = df.rename(columns=lambda x: x.strip())
    required = {"Lan Id", "Status", "Notice Sent Date", "Business"}
    if not required.issubset(df.columns):
        raise ValueError(f"LAN status Excel must have columns {required}. Found: {set(df.columns)}")
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# =========================
# LEGAL STAIRCASE
# =========================
STAIRCASE_OFFSETS = {
    "pre arbitration notice": ("Arbitration Notice", 4),
    "arbitration notice": ("Arbitral Award", 5),
    "arbitral award": ("Execution Notice", 7),
    "reminder notice": ("Legal Follow-up", 3),
    "pre-sales": ("Post Sales", 4),
    "post sales": ("Closure", 5),
}
LEGAL_INITIATED_STATUSES = {"pre arbitration notice", "arbitration notice", "arbitral award", "execution notice", "reminder notice"}

def normalize_status_fuzzy(status: str) -> str:
    status = _norm(status)
    choices = list(STAIRCASE_OFFSETS.keys())
    best_match = difflib.get_close_matches(status, choices, n=1, cutoff=0.6)
    return best_match[0] if best_match else status

def compute_next_step(last_status: str, last_date: Optional[pd.Timestamp]) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    key = normalize_status_fuzzy(last_status)
    next_info = STAIRCASE_OFFSETS.get(key)
    if not next_info or last_date is None or pd.isna(last_date):
        return None, None
    next_name, offset_days = next_info
    return next_name, last_date + timedelta(days=offset_days)

def current_legal_status(status_name: str) -> str:
    return "Legal initiated" if normalize_status_fuzzy(status_name) in LEGAL_INITIATED_STATUSES else "Pre-legal"

def summarize_lan_record(row: pd.Series) -> Dict[str, Any]:
    last_status = str(row["Status"]).strip()
    last_date = row["Notice Sent Date"]
    business = row["Business"]
    current_status_label = current_legal_status(last_status)
    next_notice, next_date = compute_next_step(last_status, last_date)
    return {
        "lan_id": row["Lan Id"],
        "business": business,
        "current_legal_status": current_status_label,
        "last_notice_name": last_status,
        "last_notice_date": None if pd.isna(last_date) else last_date.strftime("%d/%m/%Y"),
        "next_notice_name": next_notice,
        "next_notice_date": None if (next_date is None or pd.isna(next_date)) else next_date.strftime("%d/%m/%Y")
    }

def format_lan_summary(summary: Dict[str, Any]) -> str:
    return (
        f"LAN ID: {summary['lan_id']}; "
        f"Business: {summary['business']}; "
        f"Current legal status: {summary['current_legal_status']}; "
        f"Last notice sent: {summary['last_notice_name']} on {summary['last_notice_date'] or 'N/A'}; "
        f"Next notice: {summary['next_notice_name'] or 'N/A'} to be sent on {summary['next_notice_date'] or 'N/A'}"
    )

# =========================
# SYSTEM ROLE & LLM
# =========================
SYSTEM_ROLE = (
    "You are a Senior Legal Advocate with 15 years’ experience advising NBFCs on loan defaults. "
    "Keep answers concise, ≤{MAX_WORDS} words. Add up to two short practical suggestions if needed."
).replace("{MAX_WORDS}", str(MAX_WORDS))

def llm_answer(query: str, contexts: List[dict]) -> str:
    if not contexts:
        return "Not available in staircase data."
    context_text = "\n\n".join([
        f"[DOC {c['id']}] Business: {c['business']}\nQ: {c['question']}\nA: {c['answer']}"
        for c in contexts
    ])
    user_prompt = f"User Query:\n{query}\n\nRetrieved Context:\n{context_text}\n\nAnswer in ≤{MAX_WORDS} words."
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=220,
    )
    text = resp.choices[0].message.content.strip()
    if _count_words(text) > MAX_WORDS:
        text = _truncate_to_words(text, MAX_WORDS)
    return text

# =========================
# PUBLIC INTERFACE
# =========================
def answer(query: str, lan_df: Optional[pd.DataFrame] = None, force_refresh_embeddings=False) -> dict:
    df, embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE, force_refresh=force_refresh_embeddings)

    # LAN ID check
    lan_id_match = re.search(r"\b\d{3,}\b", query)
    if lan_id_match and lan_df is None:
        try:
            lan_df = load_lan_status(EXCEL_LAN_PATH)
        except Exception:
            lan_df = None
    if lan_id_match and lan_df is not None:
        lan_id = lan_id_match.group(0)
        subset = lan_df[lan_df["Lan Id"].astype(str).str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            summary = summarize_lan_record(row)
            return {"answer": format_lan_summary(summary), "contexts": [], "lan_summary": summary}

    # NBFC RAG retrieval
    ctx = retrieve(query, df, embs, top_k=TOP_K)
    out = llm_answer(query, ctx)
    return {"answer": out, "contexts": ctx}

def refresh_embeddings():
    if os.path.exists(EMBED_CACHE):
        os.remove(EMBED_CACHE)
    df, embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE, force_refresh=True)
    return {"rows": len(df), "dim": int(embs.shape[1]) if embs.size else 0}

# =========================
# STREAMLIT APP
# =========================
st.title("NBFC Legal Advocate Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Ask your question:")

if st.button("Send"):
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            result = answer(query)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as e:
            st.error(f"Error: {e}")
            with open(LOG_FILE, "a") as f:
                f.write(str(e) + "\n")

st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")