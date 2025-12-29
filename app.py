import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# CSS – SPACING FIXED
# =========================
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

.small { color:#94a3b8; font-size:0.9rem; }

.card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 22px;
}

.bento {
    background: rgba(255,255,255,0.035);
    border-radius: 18px;
    padding: 22px;
    margin: 12px 10px;
}

.answer-box {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    padding: 18px;
    border-radius: 12px;
}

.agent-box {
    background: rgba(45,212,191,0.12);
    border-left: 4px solid #2dd4bf;
    padding: 16px;
    border-radius: 12px;
}

.section-gap { margin-top: 28px; }
</style>
""", unsafe_allow_html=True)

# =========================
# OPENAI CONFIG
# =========================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_qa():
    df = pd.read_excel("legal_staircase.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    return df[["question", "answer"]]

@st.cache_data
def load_lan():
    return pd.read_excel("lan_data.xlsx", dtype=str)

qa_df = load_qa()
lan_df = load_lan()

# =========================
# EMBEDDINGS (RAG)
# =========================
def embed(texts):
    res = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return np.array([x["embedding"] for x in res["data"]])

@st.cache_data
def build_embeddings():
    return embed(qa_df["question"].tolist())

qa_emb = build_embeddings()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# =========================
# AGENT ADVICE
# =========================
def agent_advice(context):
    prompt = f"""
You are a senior NBFC collections manager.
Based ONLY on the context below, suggest 3 compliant agent actions.

Context:
{context}
"""
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return r.choices[0].message.content

# =========================
# ANSWER ENGINE (STRICT RAG)
# =========================
def answer_query(q):
    # LAN FLOW
    lan_match = re.search(r"\\b\\d{3,}\\b", q)
    if lan_match:
        lan = lan_match.group()
        row = lan_df[lan_df["Lan Id"] == lan]
        if not row.empty:
            text = row.iloc[0].to_string()
            return text, agent_advice(text)

    # LEGAL RAG
    qv = embed([q])[0]
    sims = [cosine(qv,e) for e in qa_emb]
    idx = int(np.argmax(sims))

    if max(sims) < 0.30:
        return "Not covered in current NBFC policy repository.", ""

    ans = qa_df.iloc[idx]["answer"]
    return ans, agent_advice(ans)

# =========================
# PDF
# =========================
def make_pdf(q,a,g):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    p.drawString(40,750,"NBFC Legal Intelligence Summary")
    p.drawString(40,730,f"Query: {q}")
    y = 700
    for line in a.split("\\n"):
        p.drawString(40,y,line)
        y -= 14
    p.drawString(40,y-20,"Agent Advice:")
    p.drawString(40,y-40,g[:300])
    p.save()
    buf.seek(0)
    return buf

# =========================
# HEADER
# =========================
st.markdown("<div class='hero'>Legal Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='small'>NBFC Legal Staircase • LAN Intelligence • Agent Communication</div>", unsafe_allow_html=True)

# =========================
# INTRO
# =========================
c1,c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="card">
    <b>What does this assistant do?</b>
    <ul class="small">
      <li>Explains NBFC recovery stages</li>
      <li>Interprets SARFAESI & Section 138</li>
      <li>Fetches LAN recovery status</li>
      <li>Suggests compliant agent actions</li>
    </ul>
    ⚠ Operational guidance only
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <b>How to use</b>
    <ul class="small">
      <li>Ask legal / collections question</li>
      <li>Enter LAN ID (e.g. 22222)</li>
      <li>Review system + agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# THREE BOXES
# =========================
b1,b2,b3 = st.columns(3)
with b1:
    st.markdown("<div class='bento'>⚖️ <b>Legal Staircase</b><br><span class='small'>DPD-based recovery flow</span></div>", unsafe_allow_html=True)
with b2:
    st.markdown("<div class='bento'>🔍 <b>LAN Intelligence</b><br><span class='small'>Account-level status</span></div>", unsafe_allow_html=True)
with b3:
    st.markdown("<div class='bento'>📞 <b>Communication</b><br><span class='small'>Compliant agent scripts</span></div>", unsafe_allow_html=True)

# =========================
# CHAT
# =========================
query = st.chat_input("Ask legal question or enter LAN ID")

if query:
    with st.spinner("Analyzing..."):
        ans, adv = answer_query(query)
        a,b = st.columns([2,1])

        with a:
            st.markdown("<div class='answer-box'><b>System Answer</b><br>"+ans+"</div>", unsafe_allow_html=True)

        with b:
            if adv:
                st.markdown("<div class='agent-box'><b>Agent Advice</b><br>"+adv+"</div>", unsafe_allow_html=True)

        st.download_button(
            "📄 Download Summary",
            data=make_pdf(query,ans,adv),
            file_name="nbfc_summary.pdf",
            mime="application/pdf"
        )

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style="text-align:center;color:#64748b;font-size:0.8rem;">
Designed by <b>Mohit Raheja</b> | NBFC Legal Intelligence
</p>
""", unsafe_allow_html=True)
