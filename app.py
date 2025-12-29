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
# CSS – PREMIUM APP UI
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color:#f8fafc; }
.hero { font-size:2.6rem; font-weight:800;
background:linear-gradient(90deg,#3b82f6,#2dd4bf);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub { color:#94a3b8; margin-bottom:20px; }
.card { background:rgba(255,255,255,0.04); border-radius:16px; padding:18px; }
.bento { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
border-radius:18px; padding:22px; }
.answer { background:rgba(15,23,42,0.7); border-left:4px solid #3b82f6;
border-radius:12px; padding:18px; }
.agent { background:rgba(45,212,191,0.12); border-left:4px solid #2dd4bf;
border-radius:12px; padding:16px; }
.small { font-size:0.85rem; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (SAFE)
# ======================================================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel("legal_staircase.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"question":"q","answer":"a"})
    return df[["q","a"]]

@st.cache_data
def load_lan():
    return pd.read_excel("lan_data.xlsx", dtype=str)

qa_df = load_qa()
lan_df = load_lan()

# ======================================================
# EMBEDDINGS
# ======================================================
def embed(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return np.array([d["embedding"] for d in res["data"]], dtype=np.float32)

@st.cache_data
def build_embeddings():
    return embed(qa_df["q"].tolist())

qa_emb = build_embeddings()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ======================================================
# AGENT ADVICE (MANDATORY)
# ======================================================
def agent_advice(context):
    prompt = f"""
You are a senior NBFC collections manager.
Give 3 compliant, polite agent actions based ONLY on this context:

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
# ANSWER ENGINE (STRICT RAG)
# ======================================================
def answer_query(query):
    lan = re.search(r"\b\d{3,}\b", query)

    if lan:
        row = lan_df[lan_df["Lan Id"] == lan.group()]
        if not row.empty:
            txt = row.iloc[0].to_string()
            return txt, agent_advice(txt)

    qv = embed([query])[0]
    sims = [cosine(qv,e) for e in qa_emb]
    idx = int(np.argmax(sims))

    if max(sims) < 0.30:
        return "This query is not covered in the NBFC policy repository.", ""

    ans = qa_df.iloc[idx]["a"]
    return ans, agent_advice(ans)

# ======================================================
# PDF
# ======================================================
def make_pdf(q,a,adv):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    p.drawString(40,750,"NBFC Legal Intelligence Summary")
    p.drawString(40,730,f"Query: {q}")
    p.drawString(40,700,"Answer:")
    t = p.beginText(40,680)
    for l in a.split("\n"): t.textLine(l)
    p.drawText(t)
    p.drawString(40,520,"Agent Advice:")
    p.drawString(40,500,adv[:250])
    p.save()
    buf.seek(0)
    return buf

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='hero'>Legal Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>NBFC Legal Staircase • LAN Intelligence • Agent Guidance</div>", unsafe_allow_html=True)

# ======================================================
# 🔥 THREE BOXES (THIS WAS MISSING)
# ======================================================
c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='bento'>
    ⚖️ <b>Legal Staircase</b><br>
    <span class='small'>SARFAESI, Section 138, Arbitration steps</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='bento'>
    🔍 <b>LAN Intelligence</b><br>
    <span class='small'>Notice history, recovery stage, status</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='bento'>
    📞 <b>Communication</b><br>
    <span class='small'>Compliant scripts & agent actions</span>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# INTRO + HOW TO USE
# ======================================================
i1,i2 = st.columns(2)

with i1:
    st.markdown("""
    <div class='card'>
    <b>What does this assistant do?</b>
    <ul class='small'>
    <li>Explains NBFC legal recovery stages</li>
    <li>Interprets SARFAESI & Section 138</li>
    <li>Fetches LAN-level status</li>
    <li>Suggests compliant agent actions</li>
    </ul>
    ⚠️ Operational guidance only.
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class='card'>
    <b>How to use</b>
    <ul class='small'>
    <li>Ask a legal question</li>
    <li>Enter LAN ID (e.g. 22222)</li>
    <li>Review answer & agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# CHAT
# ======================================================
q = st.chat_input("Ask legal question or enter LAN ID")

if q:
    ans, adv = answer_query(q)

    a,b = st.columns([2,1])
    with a:
        st.markdown("<div class='answer'><b>System Answer</b><br>"+ans+"</div>", unsafe_allow_html=True)
    with b:
        if adv:
            st.markdown("<div class='agent'><b>Agent Advice</b><br>"+adv+"</div>", unsafe_allow_html=True)

    st.download_button(
        "📄 Download Summary",
        data=make_pdf(q,ans,adv),
        file_name="nbfc_summary.pdf",
        mime="application/pdf"
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style='text-align:center;color:#64748b;font-size:0.8rem'>
Designed by <b>Mohit Raheja</b> | Applied AI – NBFC Collections
</p>
""", unsafe_allow_html=True)
