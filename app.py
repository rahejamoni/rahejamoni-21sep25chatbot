import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# CSS – SAFE, COMPACT, NON-BREAKING
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}
.hero {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg,#3b82f6,#2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 18px;
}
.small {
    font-size: 0.85rem;
    color: #94a3b8;
}
.response {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    padding: 16px;
    border-radius: 10px;
}
.agent {
    background: rgba(45,212,191,0.12);
    border-left: 4px solid #2dd4bf;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI (STABLE)
# ======================================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# LOAD DATA (SAFE)
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel("legal_staircase.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"question":"q", "answer":"a"})
    return df[["q","a"]]

@st.cache_data
def load_lan():
    return pd.read_excel("lan_data.xlsx", dtype=str)

qa_df = load_qa()
lan_df = load_lan()

# ======================================================
# EMBEDDINGS (RAG)
# ======================================================
@st.cache_data
def build_embeddings():
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=qa_df["q"].tolist()
    )
    return np.array([d["embedding"] for d in res["data"]])

qa_emb = build_embeddings()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ======================================================
# AGENT ADVICE (MANDATORY)
# ======================================================
def agent_advice(context):
    prompt = f"""
You are a senior NBFC collections manager.
Give 3 compliant agent actions ONLY based on this policy text.
Short, professional, actionable.

Policy:
{context}
"""
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return r.choices[0].message.content

# ======================================================
# ANSWER ENGINE (STRICT RAG)
# ======================================================
def answer_query(query):
    # LAN FLOW
    m = re.search(r"\b\d{3,}\b", query)
    if m:
        lan = m.group()
        row = lan_df[lan_df.iloc[:,0] == lan]
        if not row.empty:
            text = row.iloc[0].to_string()
            return text, agent_advice(text)

    # LEGAL FLOW
    q_vec = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )["data"][0]["embedding"]

    sims = [cosine(q_vec, e) for e in qa_emb]
    idx = int(np.argmax(sims))

    if max(sims) < 0.30:
        return "This question is not covered in the NBFC legal repository.", ""

    answer = qa_df.iloc[idx]["a"]
    return answer, agent_advice(answer)

# ======================================================
# PDF DOWNLOAD
# ======================================================
def make_pdf(q,a,adv):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    p.drawString(50,750,"NBFC Legal Intelligence Summary")
    p.drawString(50,730,f"Query: {q}")
    t = p.beginText(50,700)
    for l in a.split("\n"):
        t.textLine(l)
    p.drawText(t)
    p.drawString(50,480,"Agent Advice:")
    p.drawString(50,460,adv[:250])
    p.save()
    buf.seek(0)
    return buf

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="hero">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="small">NBFC Legal Staircase • LAN Intelligence • Agent Communication</div>', unsafe_allow_html=True)

# ======================================================
# INTRO + HOW TO USE
# ======================================================
c1,c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="card">
    <b>What does this assistant do?</b>
    <ul class="small">
    <li>Explains NBFC legal notices & recovery stages</li>
    <li>Interprets SARFAESI, Section 138, Arbitration</li>
    <li>Fetches LAN recovery status</li>
    <li>Suggests compliant agent actions</li>
    </ul>
    ⚠️ Operational guidance only.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <b>How to use</b>
    <ul class="small">
    <li>Ask legal / collections questions</li>
    <li>Enter LAN ID (e.g. 22222)</li>
    <li>Review system answer & agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# THREE FEATURE BOXES (FIXED)
# ======================================================
b1,b2,b3 = st.columns(3)
b1.markdown("<div class='card'><b>⚖️ Legal Staircase</b><p class='small'>DPD-based recovery & notice flow</p></div>", unsafe_allow_html=True)
b2.markdown("<div class='card'><b>🔍 LAN Intelligence</b><p class='small'>Account-level recovery status</p></div>", unsafe_allow_html=True)
b3.markdown("<div class='card'><b>📞 Communication</b><p class='small'>Compliant agent scripts & actions</p></div>", unsafe_allow_html=True)

# ======================================================
# CHAT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID")

if query:
    with st.spinner("Analyzing NBFC policy..."):
        ans, adv = answer_query(query)
        a,b = st.columns([2,1])
        a.markdown("<div class='response'><b>System Answer</b><br>"+ans+"</div>", unsafe_allow_html=True)
        if adv:
            b.markdown("<div class='agent'><b>Agent Advice</b><br>"+adv+"</div>", unsafe_allow_html=True)

        pdf = make_pdf(query, ans, adv)
        st.download_button("📄 Download Summary", pdf, "nbfc_summary.pdf", "application/pdf")

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style='text-align:center;color:#64748b;font-size:0.8rem'>
Designed by <b>Mohit Raheja</b> | NBFC Legal Intelligence
</p>
""", unsafe_allow_html=True)
