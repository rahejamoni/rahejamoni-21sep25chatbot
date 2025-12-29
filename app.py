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
    layout="wide"
)

# ======================================================
# GLOBAL CSS (PROFESSIONAL & COMPACT)
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.hero {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub {
    color: #94a3b8;
    font-size: 1.05rem;
}

.tag {
    display:inline-block;
    padding:4px 10px;
    border-radius:14px;
    font-size:12px;
    background:rgba(45,212,191,0.12);
    color:#2dd4bf;
    border:1px solid rgba(45,212,191,0.25);
}

.response {
    background: rgba(15,23,42,0.75);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px;
}

.agent {
    background: rgba(34,197,94,0.12);
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 14px;
}

.small {
    color:#94a3b8;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# CONFIG
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings_v2.pkl"

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# UTIL FUNCTIONS
# ======================================================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def chat(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return res.choices[0].message["content"]

def embed(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [x["embedding"] for x in res["data"]]

def is_general(q):
    keywords = ["capital", "define", "what is", "who is", "india", "delhi"]
    return any(k in q.lower() for k in keywords)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.lower()
    df["id"] = range(len(df))
    return df[["id","question","answer","business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype=str)
    df["notice sent date"] = pd.to_datetime(df["notice sent date"], errors="coerce")
    return df

def build_embeddings():
    qa = load_qa()
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE,"rb") as f:
            saved = pickle.load(f)
            if saved["len"] == len(qa):
                return qa, saved["emb"]
    corpus = [q+" || "+a for q,a in zip(qa["question"], qa["answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)
    with open(EMBED_CACHE,"wb") as f:
        pickle.dump({"emb":emb,"len":len(qa)}, f)
    return qa, emb

qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# HEADER
# ======================================================
st.markdown(f"""
<div class="card">
<div class="hero">NBFC Legal & Collections Intelligence Assistant</div>
<p class="sub">
AI-powered decision-support system for NBFC collection agents to understand
legal processes, loan status, and compliant recovery actions.
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INTRO + FEATURES (NO SCROLL WASTE)
# ======================================================
intro_l, intro_r = st.columns([2,1])

with intro_l:
    st.markdown("""
    <div class="card">
    <b>🔍 What does this assistant do?</b>
    <ul class="small">
        <li>Explains NBFC legal staircase (Demand → Pre-sale → Auction)</li>
        <li>Fetches recovery status using LAN</li>
        <li>Guides compliant customer communication</li>
    </ul>
    <p class="small">⚠️ Operational guidance only. Not legal advice.</p>
    </div>
    """, unsafe_allow_html=True)

with intro_r:
    st.markdown("""
    <div class="card">
    <b>⚡ How to use</b>
    <ul class="small">
        <li>Ask legal or collections questions</li>
        <li>Enter LAN ID (e.g. 22222)</li>
        <li>Ask general questions if required</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# THREE FEATURE BOXES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <h4>⚖️ Legal Staircase</h4>
    <p class="small">Interpret SARFAESI, Sec 138 & Arbitration steps</p>
    <span class="tag">RBI 2024</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h4>🔍 LAN Intelligence</h4>
    <p class="small">Notice history, status & recovery stage</p>
    <span class="tag">Live Data</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h4>📞 Communication</h4>
    <p class="small">Polite, compliant agent call guidance</p>
    <span class="tag">Audit Safe</span>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# CHAT (VISIBLE WITHOUT SCROLL)
# ======================================================
st.markdown("### 💬 Ask a Question")
query = st.text_input(
    "",
    placeholder="Ask a legal question or enter LAN ID",
    label_visibility="collapsed"
)

if st.button("🚀 Submit") and query:
    # LAN check
    lan_match = re.search(r"\b\d{3,}\b", query)
    agent_tips = ""

    if lan_match:
        lan = lan_match.group(0)
        row = lan_df[lan_df["lan id"] == lan]
        if not row.empty:
            r = row.iloc[0]
            answer = f"LAN {lan} status is **{r['status']}**, notice sent on **{r['notice sent date'].date()}**."
            agent_tips = chat(f"Give 3 polite collection call suggestions for: {answer}")
        else:
            answer = "LAN not found in system."

    elif is_general(query):
        answer = chat(query)

    else:
        qv = embed([query])[0]
        sims = [cosine(qv, e) for e in qa_emb]
        if max(sims) < 0.35:
            answer = chat(query)
        else:
            best = qa_df.iloc[int(np.argmax(sims))]["answer"]
            answer = best
            agent_tips = chat(f"Give 3 compliant agent suggestions for: {best}")

    # OUTPUT (NO SCROLL)
    out_l, out_r = st.columns([2,1])

    with out_l:
        st.markdown("<div class='response'><b>🧠 System Response</b><br><br>"+answer+"</div>", unsafe_allow_html=True)

    with out_r:
        if agent_tips:
            st.markdown("<div class='agent'><b>🎧 Agent Suggestions</b><br><br>"+agent_tips+"</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p class="small" style="text-align:center">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence
</p>
""", unsafe_allow_html=True)
