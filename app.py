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
# API KEY (SAFE)
# ======================================================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# CSS – COMPACT ENTERPRISE UI
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color: #f8fafc; }
.hero { font-size:2.6rem; font-weight:800;
background:linear-gradient(90deg,#3b82f6,#2dd4bf);
-webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.sub { color:#94a3b8;font-size:1rem;margin-bottom:18px; }

.card { background:rgba(255,255,255,0.04);border-radius:14px;padding:16px;margin-bottom:14px; }
.box { background:rgba(255,255,255,0.03);border-radius:16px;padding:20px; }
.response { background:rgba(15,23,42,0.7);border-left:4px solid #3b82f6;padding:14px;border-radius:10px; }
.agent { background:rgba(45,212,191,0.12);border-left:4px solid #2dd4bf;padding:14px;border-radius:10px; }
.small { color:#94a3b8;font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

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
    df = pd.read_excel("lan_data.xlsx", dtype=str)
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
    return np.array([d["embedding"] for d in res["data"]])

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
Based ONLY on the policy text below, suggest 3 compliant agent actions.
Keep concise and professional.

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
def answer_query(q):
    # LAN FLOW
    m = re.search(r"\b\d{3,}\b", q)
    if m:
        lan = m.group(0)
        r = lan_df[lan_df["Lan Id"] == lan]
        if not r.empty:
            txt = f"""
LAN ID: {lan}
Status: {r.iloc[0]['Status']}
Business: {r.iloc[0]['Business']}
"""
            return txt.strip(), agent_advice(txt)

    # LEGAL RAG FLOW
    q_vec = embed([q])[0]
    sims = [cosine(q_vec,e) for e in qa_emb]
    idx = int(np.argmax(sims))

    if max(sims) < 0.30:
        return "This question is not covered in the current NBFC policy repository.", ""

    ans = qa_df.iloc[idx]["a"]
    return ans, agent_advice(ans)

# ======================================================
# PDF EXPORT
# ======================================================
def generate_pdf(q,a,g):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    p.drawString(40,750,"NBFC Legal Intelligence Summary")
    p.drawString(40,730,f"Query: {q}")
    p.drawString(40,700,"Answer:")
    t = p.beginText(40,680)
    for l in a.split("\n"): t.textLine(l)
    p.drawText(t)
    p.drawString(40,500,"Agent Advice:")
    p.drawString(40,480,g[:200])
    p.save()
    buf.seek(0)
    return buf

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='hero'>Legal Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>NBFC Legal Staircase • LAN Intelligence • Agent Communication</div>", unsafe_allow_html=True)

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
    <li>Interprets SARFAESI, Section 138 & arbitration</li>
    <li>Fetches LAN-level recovery status</li>
    <li>Suggests compliant agent communication</li>
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
    <li>Review answer & agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# THREE FEATURE BOXES
# ======================================================
b1,b2,b3 = st.columns(3)
b1.markdown("<div class='box'><b>⚖️ Legal Staircase</b><p class='small'>SARFAESI, Section 138, Arbitration stages</p></div>", unsafe_allow_html=True)
b2.markdown("<div class='box'><b>🔍 LAN Intelligence</b><p class='small'>Loan-level status & notices</p></div>", unsafe_allow_html=True)
b3.markdown("<div class='box'><b>📞 Communication</b><p class='small'>Audit-safe agent scripts</p></div>", unsafe_allow_html=True)

# ======================================================
# CHAT
# ======================================================
query = st.chat_input("Ask legal question or enter LAN ID")

if query:
    with st.spinner("Analyzing policy..."):
        ans, adv = answer_query(query)
        ca,cb = st.columns([2,1])

        ca.markdown(f"<div class='response'><b>System Answer</b><br>{ans}</div>", unsafe_allow_html=True)
        if adv:
            cb.markdown(f"<div class='agent'><b>Agent Advice</b><br>{adv}</div>", unsafe_allow_html=True)

        pdf = generate_pdf(query, ans, adv)
        st.download_button("📄 Download Summary", pdf, "nbfc_legal_summary.pdf", "application/pdf")

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center;color:#64748b;font-size:0.8rem;">
Designed by <b>Mohit Raheja</b> | Applied AI – NBFC Collections
</p>
""", unsafe_allow_html=True)
