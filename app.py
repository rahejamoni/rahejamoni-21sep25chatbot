import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal & Collections Intelligence Assistant",
    page_icon="📘",
    layout="wide"
)

# ======================================================
# BASIC CSS (VISIBLE UI CHANGE)
# ======================================================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.small-text {
    color: #9ba3af;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
<h1>📘 NBFC Legal & Collections Intelligence Assistant</h1>
<p class="small-text">
AI-powered decision-support system for NBFC collection agents to understand
legal processes, loan status, and compliant recovery actions
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INTRODUCTION
# ======================================================
st.markdown("""
<div class="card">
<h3>🔍 What does this assistant do?</h3>
<ul class="small-text">
<li>Explains NBFC legal notices (Pre-sale, Auction, Possession, etc.)</li>
<li>Identifies recovery status using Loan Account Number (LAN)</li>
<li>Guides agents on compliance timelines</li>
<li>Suggests polite, compliant customer communication</li>
</ul>
<p class="small-text">⚠️ This tool assists agents and does not replace legal advice.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CONFIG
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings_v2.pkl"

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# UTILITIES
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message["content"].strip()

def embed(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in response["data"]]

# ======================================================
# LOAD LEGAL QA (SAFE)
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "question": "Question",
        "questions": "Question",
        "answer": "Answer",
        "answers": "Answer",
        "business": "Business",
        "vertical": "Business"
    }
    df = df.rename(columns=rename_map)

    required = {"Question", "Answer", "Business"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in legal_staircase.xlsx: {missing}")
        st.stop()

    df["id"] = range(len(df))
    return df[["id", "Question", "Answer", "Business"]]

# ======================================================
# LOAD LAN DATA (SAFE)
# ======================================================
@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype=str)
    df.columns = df.columns.str.strip()

    required = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in lan_data.xlsx: {missing}")
        st.stop()

    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(
        df["Notice Sent Date"], errors="coerce", dayfirst=True
    )
    return df

# ======================================================
# BUILD / LOAD EMBEDDINGS (SAFE CACHE)
# ======================================================
def build_embeddings():
    qa_df = load_qa()

    if os.path.exists(EMBED_CACHE):
        try:
            with open(EMBED_CACHE, "rb") as f:
                saved = pickle.load(f)
            if isinstance(saved, dict) and "emb" in saved:
                if saved.get("len") == len(qa_df):
                    return qa_df, saved["emb"]
        except Exception:
            pass

    corpus = [q + " || " + a for q, a in zip(qa_df["Question"], qa_df["Answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"emb": emb, "len": len(qa_df)}, f)

    return qa_df, emb

# ======================================================
# MAIN ANSWER LOGIC
# ======================================================
def answer_query(query):
    # LAN lookup
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            date = r["Notice Sent Date"]
            d = date.strftime("%d-%m-%Y") if pd.notna(date) else "N/A"

            answer = (
                f"LAN {lan_id} belongs to **{r['Business']}** vertical. "
                f"Current status is **{r['Status']}**, notice sent on **{d}**."
            )

            tips = chat(
                f"Give 3 polite, compliant NBFC collection call suggestions for this case:\n{answer}"
            )
            return answer, tips

    # Legal Q&A
    q_vec = embed([query])[0]
    sims = [cosine(q_vec, e) for e in qa_emb]
    best_answer = qa_df.iloc[int(np.argmax(sims))]["Answer"]

    tips = chat(
        f"Give 3 polite NBFC collection call suggestions using this context:\n{best_answer}"
    )
    return best_answer, tips

# ======================================================
# LOAD DATA
# ======================================================
qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# INPUT
# ======================================================
st.markdown('<div class="card"><h3>💬 Ask a Question</h3></div>', unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="e.g. What is a pre-sale notice? | Enter LAN ID"
)

if st.button("🚀 Submit"):
    if query.strip():
        answer, tips = answer_query(query)

        st.markdown('<div class="card"><h3>🧠 System Response</h3></div>', unsafe_allow_html=True)
        st.success(answer)

        st.markdown('<div class="card"><h3>🎧 Agent Compliance Suggestions</h3></div>', unsafe_allow_html=True)
        st.warning(tips)
