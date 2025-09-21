# rag_nbfc_legal_bot.py
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
import streamlit as st
from openai import OpenAI

# ========= LOAD SECRETS =========
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EXCEL_QA_PATH = st.secrets["EXCEL_QA_PATH"]   # legal_staircase.xlsx
EXCEL_LAN_PATH = st.secrets["EXCEL_LAN_PATH"] # lan_data.xlsx
LOG_FILE = st.secrets["LOG_FILE"]              # error_log.txt

# ========= CONFIG =========
EMBED_CACHE = "embeddings.pkl"  # This will be auto-created in the GitHub repo
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_WORDS = 150

# ========= API CLIENT =========
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= UTILITIES =========
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def _count_words(text: str) -> int:
    return len(text.strip().split())

def _truncate_to_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words])

# ========= LOAD DATA =========
def load_qa(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df.rename(columns={
        "questions": "Question",
        "answers": "Answer",
        "Business": "Business"
    })
    if not {"Question", "Answer"}.issubset(df.columns):
        raise ValueError(f"Excel must have columns 'Question' and 'Answer'. Found: {set(df.columns)}")
    df["id"] = np.arange(len(df))
    return df[["id", "Question", "Answer", "Business"]]

# ========= EMBEDDING FUNCTIONS =========
def embed_texts(texts: List[str]) -> List[List[float]]:
    BATCH = 128
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def build_or_load_embeddings(excel_path=EXCEL_QA_PATH, cache_path=EMBED_CACHE) -> Tuple[pd.DataFrame, np.ndarray]:
    # Load cached embeddings if present
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        df = load_qa(excel_path)
        if saved.get("csv_len") == len(df):
            return saved["df"], saved["embeddings"]

    # Build embeddings if cache doesn't exist or is outdated
    df = load_qa(excel_path)
    corpus = [f"Business: {b}\nQ: {q}\nA: {a}" for q, a, b in zip(df["Question"], df["Answer"], df["Business"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)

    return df, vecs

# ========= RETRIEVAL =========
def retrieve(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = TOP_K) -> List[dict]:
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

# ========= SYSTEM ROLE =========
SYSTEM_ROLE = (
    "You are a Senior Legal Advocate with 15 years’ experience advising NBFCs on loan defaults. "
    "You deeply understand legal staircases, notices, arbitration, repossession, SARFAESI, and execution. "
    "Advise NBFC business/legal teams on what to send, when, and why. "
    f"Be clear, professional, and keep the answer within {MAX_WORDS} words. "
    "If recommendations are asked, add up to two ultra-brief, practical suggestions."
)

# ========= LLM ANSWER =========
def llm_answer(query: str, contexts: List[dict]) -> str:
    if not contexts:
        return "Not available in staircase data."
    context_text = "\n\n".join([
        f"[DOC {c['id']}] Business: {c['business']}\nQ: {c['question']}\nA: {c['answer']}"
        for c in contexts
    ])
    user_prompt = (
        f"Query:\n{query}\n\n"
        "Retrieved Context (use strictly, do not invent):\n"
        f"{context_text}\n\n"
        f"Answer in ≤{MAX_WORDS} words."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=220,
    )
    text = resp.choices[0].message.content.strip()
    if _count_words(text) > MAX_WORDS:
        text = _truncate_to_words(text, MAX_WORDS)
    return text

# ========= MAIN PUBLIC FUNCTIONS =========
def answer(query: str) -> dict:
    df, embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
    ctx = retrieve(query, df, embs, top_k=TOP_K)
    out = llm_answer(query, ctx)
    return {"answer": out, "contexts": ctx}

def refresh_embeddings():
    if os.path.exists(EMBED_CACHE):
        os.remove(EMBED_CACHE)
    df, embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
    return {"rows": len(df), "dim": int(embs.shape[1]) if embs.size else 0}
