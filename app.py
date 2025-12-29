import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Intel | Legal & Collections",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# CSS (Premium Enterprise UI)
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color:#f8fafc; }
.bento-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 22px;
}
.hero-text {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg,#3b82f6,#2dd4bf);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.sub-text { color:#94a3b8; font-size:1.1rem; }
.response-box {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
}
.agent-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px;
    padding: 16px;
}
.small { color:#94a3b8; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (v0.28)
# ======================================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# FILE PATHS
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings.pkl"

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_text(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in res["data"]]

def agent_advice(context):
    prompt = f"""
You are a senior NBFC collections strategist.
Based ONLY on the context below, give 3 short compliant calling suggestions.

Context:
{context}

Format:
- ...
- ...
- ...
"""
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return res.choices[0].message["content"]

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "question":"Question",
        "questions":"Question",
        "answer":"Answer",
        "answers":"Answer",
        "business":"Business"
    })
    df["id"] = range(len(df))
    return df[["id","Question","Answer","Business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype=str)
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], errors="coerce")
    return df

def build_embeddings():
    qa_df = load_qa()
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE,"rb") as f:
            saved = pickle.load(f)
        if saved["len"] == len(qa_df):
            return qa_df, saved["emb"]

    corpus = [q+" || "+a for q,a in zip(qa_df["Question"],qa_df["Answer"])]
    emb = np.array(embed_text(corpus))
    with open(EMBED_CACHE,"wb") as f:
        pickle.dump({"emb":emb,"len":len(qa_df)}, f)
    return qa_df, emb

qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# CORE RAG LOGIC
# ======================================================
def answer_query(query):
    # 1️⃣ LAN DETECTION
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group()
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            text = f"""
LAN {lan_id}
Business: {r['Business']}
Status: {r['Status']}
Notice Date: {r['Notice Sent Date']}
"""
            return text, agent_advice(text)

    # 2️⃣ LEGAL RAG
    q_vec = embed_text([query])[0]
    sims = [cosine(q_vec, e) for e in qa_emb]
    best_idx = int(np.argmax(sims))
    score = max(sims)

    if score < 0.30:
        return "No exact legal step found in internal knowledge base.", ""

    answer = qa_df.iloc[best_idx]["Answer"]
    advice = agent_advice(answer)
    return answer, advice

# ======================================================
# UI HEADER
# ======================================================
st.markdown('<h1 class="hero-text">Legal Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">NBFC collections & legal decision support (RAG-based)</p>', unsafe_allow_html=True)

# ======================================================
# FEATURE BOXES
# ======================================================
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="bento-card"><h4>⚖️ Legal Staircase</h4><p class="small">Excel-driven legal steps (SARFAESI, Sec 138, Arbitration)</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="bento-card"><h4>🔍 LAN Intelligence</h4><p class="small">LAN-level notice and recovery insights</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="bento-card"><h4>📞 Agent Guidance</h4><p class="small">Compliant calling advice generated per case</p></div>', unsafe_allow_html=True)

# ======================================================
# CHAT INPUT
# ======================================================
query = st.chat_input("Ask legal question or enter LAN ID...")

if query:
    with st.spinner("Analyzing through Legal Intelligence Engine..."):
        answer, advice = answer_query(query)

    colA, colB = st.columns([2,1])

    with colA:
        st.markdown("### 💡 Legal / System Answer")
        st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)

    with colB:
        if advice:
            st.markdown("### 🎧 Agent Advice")
            st.markdown(f'<div class="agent-box">{advice}</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center;color:#64748b;font-size:0.85rem;">
Created by <b>Mohit Raheja</b> | RAG-based Legal Intelligence System
</p>
""", unsafe_allow_html=True)
