import os
import re
import pickle
import difflib
from datetime import timedelta
from typing import List, Dict, Any, Optional

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

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# CONFIG
# =========================
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_WORDS = 150

# =========================
# UTIL FUNCTIONS
# =========================
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def _truncate(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_WORDS])

# =========================
# LOAD QA EXCEL (staircase)
# =========================
@st.cache_data
def load_qa(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    rename_map = {
        "questions": "Question",
        "anwers": "Answer",
        "answers": "Answer",
        "Business": "Business"
    }
    df = df.rename(columns=rename_map)

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    required = {"id", "Business", "Question", "Answer"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns in QA file: {missing}")
        st.stop()

    return df[["id", "Business", "Question", "Answer"]]

# =========================
# LOAD LAN DATA
# =========================
@st.cache_data
def load_lan_status(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()

    req = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    if not req.issubset(df.columns):
        st.error(f"LAN Excel missing columns: {req - set(df.columns)}")
        st.stop()

    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")

    return df

# =========================
# EMBEDDINGS
# =========================
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_or_load_embeddings(path: str, cache_path: str):
    df = load_qa(path)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        if saved["csv_len"] == len(df):
            return saved["df"], saved["embeddings"]

    corpus = [f"Business: {b}\nQ: {q}\nA: {a}"
              for q, a, b in zip(df["Question"], df["Answer"], df["Business"])]

    vecs = np.array(embed_texts(corpus), dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)

    return df, vecs

# =========================
# RAG RETRIEVAL
# =========================
def retrieve(query: str, df: pd.DataFrame, embeddings: np.ndarray):
    q_vec = np.array(embed_texts([query])[0])
    sims = np.array([_cosine_sim(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:TOP_K]

    return [
        {
            "id": int(df.iloc[i]["id"]),
            "question": df.iloc[i]["Question"],
            "answer": df.iloc[i]["Answer"],
            "business": df.iloc[i]["Business"],
            "score": float(sims[i]),
        }
        for i in top_idx
    ]

# =========================
# LAN STATUS SUMMARY
# =========================
def summarize_lan_record(row: pd.Series) -> str:
    last_date = row["Notice Sent Date"]
    last_date_str = last_date.strftime("%d/%m/%Y") if pd.notna(last_date) else "N/A"

    return (
        f"LAN ID: {row['Lan Id']}\n"
        f"Business: {row['Business']}\n"
        f"Current Status: {row['Status']}\n"
        f"Last Notice Date: {last_date_str}"
    )

# =========================
# LLM ANSWER
# =========================
def llm_answer(query: str, ctx: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join([
        f"[{c['id']}] Business: {c['business']}\nQ: {c['question']}\nA: {c['answer']}"
        for c in ctx
    ])

    prompt = f"""
Answer ONLY using the information provided below from the Excel staircase file.

User query: {query}

Context:
{context_text}

Keep answer under {MAX_WORDS} words.
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": "You answer strictly from the Excel file."},
                  {"role": "user", "content": prompt}],
        temperature=0.2
    )

    return _truncate(resp.choices[0].message.content.strip())

# =========================
# MAIN ANSWER FUNCTION
# =========================
def answer(query: str, lan_df: pd.DataFrame, qa_df: pd.DataFrame, qa_embs: np.ndarray):
    # 1️⃣ Check LAN ID
    lan_id_match = re.search(r"\b\d{3,}\b", query)
    if lan_id_match:
        lan_id = lan_id_match.group(0)
        subset = lan_df[lan_df["Lan Id"] == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            return summarize_lan_record(row)

    # 2️⃣ Otherwise → Use RAG
    ctx = retrieve(query, qa_df, qa_embs)
    if ctx:
        return llm_answer(query, ctx)

    return "No matching information found in the legal staircase file."

# =========================
# LOAD DATA ONCE
# =========================
qa_df, qa_embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
lan_df = load_lan_status(EXCEL_LAN_PATH)

# =========================
# STREAMLIT UI
# =========================
st.title("NBFC Legal Advocate Bot 🤖")
st.write("Ask about notices, legal staircase, or LAN status!")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Enter your question:")

if st.button("Send"):
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        result = answer(query, lan_df, qa_df, qa_embs)
        st.session_state.messages.append({"role": "assistant", "content": result})

# Chat history
st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
