import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Intel | Legal & Collections",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# CSS – COMPACT, APP-LIKE UI
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color: #f8fafc; }
.card { background: rgba(255,255,255,0.04); border-radius:14px; padding:18px; margin-bottom:16px; }
.small { color:#94a3b8; font-size:0.85rem; }
.hero { font-size:2.5rem; font-weight:800; background:linear-gradient(90deg,#3b82f6,#2dd4bf); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.response { background: rgba(15,23,42,0.7); border-left:4px solid #3b82f6; padding:16px; border-radius:10px; }
.agent { background: rgba(45,212,191,0.1); border-left:4px solid #2dd4bf; padding:14px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (STABLE)
# ======================================================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel("legal_staircase.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "question": "Question",
        "questions": "Question",
        "answer": "Answer",
        "answers": "Answer"
    })
    df["id"] = range(len(df))
    return df[["id", "Question", "Answer"]]

@st.cache_data
def load_lan():
    df = pd.read_excel("lan_data.xlsx", dtype=str)
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], errors="coerce")
    return df

qa_df = load_qa()
lan_df = load_lan()

# ======================================================
# EMBEDDINGS (RAG)
# ======================================================
def embed(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in res["data"]]

@st.cache_data
def build_embeddings():
    corpus = qa_df["Question"].tolist()
    vectors = embed(corpus)
    return np.array(vectors, dtype=np.float32)

qa_emb = build_embeddings()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ======================================================
# AGENT ADVICE GENERATOR (MANDATORY)
# ======================================================
def agent_advice(context):
    prompt = f"""
You are a senior NBFC collections manager.
Based ONLY on the following policy context, suggest 3 compliant call actions.
Keep it short, polite, and actionable.

Policy context:
{context}
"""
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return res.choices[0].message["content"]

# ======================================================
# MAIN ANSWER ENGINE (STRICT RAG)
# ======================================================
def answer_query(query):
    # 1️⃣ LAN FLOW
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan]
        if not row.empty:
            r = row.iloc[0]
            text = f"""
LAN {lan}
Business: {r['Business']}
Status: {r['Status']}
Notice Date: {r['Notice Sent Date']}
"""
            return text.strip(), agent_advice(text)

    # 2️⃣ LEGAL RAG FLOW
    q_vec = embed([query])[0]
    sims = [cosine(q_vec, e) for e in qa_emb]
    best_idx = int(np.argmax(sims))
    best_score = max(sims)

    # Confidence gate – NO hallucination
    if best_score < 0.30:
        return "This query is not covered in the current NBFC legal policy repository.", ""

    answer = qa_df.iloc[best_idx]["Answer"]
    return answer, agent_advice(answer)

# ======================================================
# PDF DOWNLOAD
# ======================================================
def generate_pdf(query, answer, advice):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(50, 750, "NBFC Legal Intelligence Summary")
    p.drawString(50, 730, f"Query: {query}")
    p.drawString(50, 700, "Answer:")
    text = p.beginText(50, 680)
    for line in answer.split("\n"):
        text.textLine(line)
    p.drawText(text)
    p.drawString(50, 500, "Agent Advice:")
    p.drawString(50, 480, advice[:200])
    p.save()
    buffer.seek(0)
    return buffer

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="hero">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="small">NBFC Legal Staircase • LAN Intelligence • Agent Communication</div>', unsafe_allow_html=True)

# ======================================================
# INTRO + HOW TO USE
# ======================================================
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="card">
    <b>What does this assistant do?</b>
    <ul class="small">
        <li>Explains NBFC legal notices & recovery stages</li>
        <li>Interprets SARFAESI, Section 138 & arbitration</li>
        <li>Fetches LAN-level recovery status</li>
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
        <li>Ask a legal / collections question</li>
        <li>Enter LAN ID (e.g. 22222)</li>
        <li>Review answer & agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# CHAT INPUT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID")

if query:
    with st.spinner("Analyzing policy..."):
        answer, advice = answer_query(query)

        colA, colB = st.columns([2,1])

        with colA:
            st.markdown("<div class='response'><b>System Answer</b><br>"+answer+"</div>", unsafe_allow_html=True)

        with colB:
            if advice:
                st.markdown("<div class='agent'><b>Agent Guidance</b><br>"+advice+"</div>", unsafe_allow_html=True)

        pdf = generate_pdf(query, answer, advice)
        st.download_button(
            "📄 Download Summary",
            data=pdf,
            file_name="nbfc_legal_summary.pdf",
            mime="application/pdf"
        )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center;color:#64748b;font-size:0.8rem;">
Designed by <b>Mohit Raheja</b> | Applied AI – NBFC Collections
</p>
""", unsafe_allow_html=True)
