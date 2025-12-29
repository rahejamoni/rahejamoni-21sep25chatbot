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
    page_title="NBFC Legal & Collections Intelligence Assistant",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# ADVANCED UI CSS
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}
.bento-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.hero-text {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    color: #94a3b8;
    font-size: 1.05rem;
}
.response-box {
    background: rgba(15,23,42,0.75);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 18px;
}
.agent-box {
    background: rgba(45,212,191,0.08);
    border-left: 4px solid #2dd4bf;
    border-radius: 12px;
    padding: 14px;
}
.small {
    font-size: 0.9rem;
    color: #cbd5f5;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (stable)
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
# HELPERS
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def chat(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=250
    )
    return resp.choices[0].message["content"].strip()

def embed(texts):
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in resp["data"]]

def is_general_question(q):
    keywords = ["capital","define","what is","who is","history","country","india"]
    ql = q.lower()
    return any(k in ql for k in keywords)

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
        try:
            with open(EMBED_CACHE,"rb") as f:
                saved = pickle.load(f)
            if saved.get("len")==len(qa_df):
                return qa_df, saved["emb"]
        except:
            pass

    corpus = [q+" || "+a for q,a in zip(qa_df["Question"],qa_df["Answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)
    with open(EMBED_CACHE,"wb") as f:
        pickle.dump({"emb":emb,"len":len(qa_df)},f)
    return qa_df, emb

qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# CORE ANSWER LOGIC
# ======================================================
def answer_query(query):
    # LAN routing
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"]==lan_id]
        if not row.empty:
            r = row.iloc[0]
            ans = f"LAN {lan_id} is in **{r['Status']}** stage under **{r['Business']}**."
            tips = chat(f"Give 3 compliant collection call suggestions for:\n{ans}")
            return ans, tips

    # General knowledge
    if is_general_question(query):
        return chat(query), ""

    # RAG
    q_vec = embed([query])[0]
    sims = [cosine(q_vec,e) for e in qa_emb]
    best_idx = int(np.argmax(sims))

    if max(sims)<0.35:
        return chat(query), ""

    best_ans = qa_df.iloc[best_idx]["Answer"]
    tips = chat(f"Give 3 compliant agent suggestions based on:\n{best_ans}")
    return best_ans, tips

# ======================================================
# HERO
# ======================================================
st.markdown('<div class="hero-text">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Decision-support system for NBFC collections & compliance</p>', unsafe_allow_html=True)

# ======================================================
# INTRO + HOW TO USE
# ======================================================
i1,i2 = st.columns([2,1])
with i1:
    st.markdown("""
    <div class="bento-card">
    <b>What does this assistant do?</b>
    <ul class="small">
    <li>Explains NBFC legal notices & recovery stages</li>
    <li>Interprets SARFAESI, Section 138 & arbitration</li>
    <li>Fetches LAN-level recovery status</li>
    <li>Suggests compliant customer communication</li>
    </ul>
    ⚠️ For operational guidance only. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="bento-card">
    <b>ℹ️ How to use</b>
    <ul class="small">
    <li>Ask a legal / collections question</li>
    <li>Enter a LAN ID (e.g. 22222)</li>
    <li>Review system response & agent suggestions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# THREE CORE BOXES
# ======================================================
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="bento-card"><b>⚖️ Legal Staircase</b><br><span class="small">Structured recovery steps & notices</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="bento-card"><b>🔍 LAN Intelligence</b><br><span class="small">Account-level recovery insights</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="bento-card"><b>📞 Communication</b><br><span class="small">Audit-safe agent scripts</span></div>', unsafe_allow_html=True)

# ======================================================
# CHAT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID")

if query:
    with st.spinner("Analyzing..."):
        answer, tips = answer_query(query)

    colA,colB = st.columns([2,1])
    with colA:
        st.markdown('<div class="response-box"><b>💡 System Response</b><br><br>'+answer+'</div>', unsafe_allow_html=True)
    with colB:
        if tips:
            st.markdown('<div class="agent-box"><b>🎧 Agent Suggestions</b><br>'+tips+'</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#94a3b8; font-size:0.85rem;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence
</p>
""", unsafe_allow_html=True)
