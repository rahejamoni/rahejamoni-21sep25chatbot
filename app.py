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
    page_title="NBFC Legal & Collections AI Assistant",
    page_icon="📘",
    layout="wide"
)

# ======================================================
# COMPACT CSS (NO SCROLL, APP-LIKE)
# ======================================================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
h1 { font-size: 28px; }
.card {
    background-color: #161b22;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.small { font-size: 13.5px; color: #9ba3af; }
.section { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
input { font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER (COMPACT)
# ======================================================
st.markdown("""
<div class="card">
<h1>📘 NBFC Legal & Collections Intelligence Assistant</h1>
<p class="small">
AI decision-support tool for NBFC collection agents to understand legal processes,
loan status, and compliant recovery actions
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CONFIG
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings_v2.pkl"

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing in Secrets")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# UTIL FUNCTIONS
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def chat(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=180
    )
    return res.choices[0].message["content"].strip()

def embed(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in res["data"]]

def is_general_question(q):
    keys = ["capital", "define", "what is", "who is", "india"]
    q = q.lower()
    return any(k in q for k in keys)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "question": "Question",
        "questions": "Question",
        "answer": "Answer",
        "answers": "Answer",
        "business": "Business"
    })
    df["id"] = range(len(df))
    return df[["id", "Question", "Answer", "Business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype=str)
    df["Notice Sent Date"] = pd.to_datetime(
        df["Notice Sent Date"], errors="coerce", dayfirst=True
    )
    return df

def build_embeddings():
    qa = load_qa()
    if os.path.exists(EMBED_CACHE):
        try:
            with open(EMBED_CACHE, "rb") as f:
                saved = pickle.load(f)
            if saved.get("len") == len(qa):
                return qa, saved["emb"]
        except:
            pass

    corpus = [q + " || " + a for q, a in zip(qa["Question"], qa["Answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"emb": emb, "len": len(qa)}, f)

    return qa, emb

qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# INFO ROW (COMPACT)
# ======================================================
info_l, info_r = st.columns([2.2, 1])

with info_l:
    st.markdown("""
    <div class="card">
    <div class="section">🔍 What does this assistant do?</div>
    <ul class="small">
        <li>Explains NBFC legal notices & compliance steps</li>
        <li>Identifies recovery status using LAN</li>
        <li>Guides compliant customer communication</li>
    </ul>
    <span class="small">⚠️ For operational guidance only</span>
    </div>
    """, unsafe_allow_html=True)

with info_r:
    st.markdown("""
    <div class="card">
    <div class="section">⚡ Quick guide</div>
    <ul class="small">
        <li>Ask legal / collection questions</li>
        <li>Enter LAN ID (e.g. 22222)</li>
        <li>Ask general questions if needed</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ANSWER LOGIC
# ======================================================
def answer_query(q):
    lan_match = re.search(r"\b\d{3,}\b", q)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            d = r["Notice Sent Date"]
            d = d.strftime("%d-%m-%Y") if pd.notna(d) else "N/A"
            ans = (
                f"LAN {lan_id} → {r['Business']} | Status: {r['Status']} | Notice: {d}"
            )
            tips = chat(f"Give 2 polite NBFC agent call suggestions:\n{ans}")
            return ans, tips

    if is_general_question(q):
        return chat(q), ""

    qv = embed([q])[0]
    sims = [cosine(qv, e) for e in qa_emb]
    best = max(sims)

    if best < 0.35:
        return chat(q), ""

    ans = qa_df.iloc[int(np.argmax(sims))]["Answer"]
    tips = chat(f"Give 2 compliant agent call suggestions:\n{ans}")
    return ans, tips

# ======================================================
# ASK QUESTION (CHAT FIRST)
# ======================================================
st.markdown('<div class="card"><div class="section">💬 Ask a Question</div></div>', unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="Type a question or enter LAN ID (e.g. 22222)",
    label_visibility="collapsed"
)

if st.button("🚀 Submit"):
    if query.strip():
        answer, tips = answer_query(query)

        st.success(answer)

        if tips:
            st.info(tips)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p class="small" style="text-align:center;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence
</p>
""", unsafe_allow_html=True)
