import os
import re
import pickle
import io
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from reportlab.lib.pagesizes import A4
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
# OPENAI CONFIG
# ======================================================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# FILES
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings_v3.pkl"

# ======================================================
# STYLING
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #0f172a, #020617); color: #f8fafc; }
.card { background: rgba(255,255,255,0.04); padding: 18px; border-radius: 14px; margin-bottom: 16px; }
.small { color:#94a3b8; font-size:14px; }
.highlight { border-left:4px solid #3b82f6; padding-left:14px; }
.action { background: rgba(34,197,94,0.12); padding:12px; border-radius:10px; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("## Intel Center")

    st.markdown("""
    <div class="small">
    <b>What does this assistant do?</b><br><br>
    • Explains NBFC legal notices and recovery stages<br>
    • Interprets SARFAESI, Section 138 & arbitration steps<br>
    • Fetches LAN-level recovery status<br>
    • Suggests compliant customer communication<br><br>
    ⚠️ For operational guidance only. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="small">
    <b>ℹ️ How to use</b><br><br>
    • Ask a legal or collections question<br>
    • Enter a LAN ID (e.g. 22222)<br>
    • Review system response and agent suggestions
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
<h1 style="color:#60a5fa;">Legal Intelligence Hub</h1>
<p class="small">Advanced decision-support for NBFC collections and legal risk mitigation.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# FEATURE BOXES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <h4>⚖️ Legal Staircase</h4>
    <p class="small">NBFC recovery lifecycle: Demand → Possession → Auction → Legal.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h4>🔍 LAN Intelligence</h4>
    <p class="small">Fetch notice status, business vertical, and recovery stage using LAN.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h4>📞 Communication</h4>
    <p class="small">AI-generated compliant agent scripts & action guidance.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# DATA LOAD
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_data
def load_lan():
    return pd.read_excel(LAN_FILE, dtype=str)

def embed(texts):
    res = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
    return [d["embedding"] for d in res["data"]]

def chat(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return res.choices[0].message["content"]

qa_df = load_qa()
lan_df = load_lan()

# ======================================================
# ROUTING LOGIC (FIXES STAIRCASE BUG)
# ======================================================
def is_general_finance(q):
    keywords = ["capital", "economy", "consumer finance", "credit", "banking"]
    return any(k in q.lower() for k in keywords)

def answer_query(query):
    # LAN
    m = re.search(r"\b\d{3,}\b", query)
    if m:
        row = lan_df[lan_df["Lan Id"] == m.group()]
        if not row.empty:
            r = row.iloc[0]
            ans = f"LAN {m.group()} is in **{r['Status']}** stage under **{r['Business']}**."
            tips = chat(f"Give 3 compliant agent suggestions for: {ans}")
            return ans, tips

    # Legal staircase (Excel)
    if "staircase" in query.lower() or "sarfaesi" in query.lower():
        return qa_df.iloc[0]["answer"], chat("Give agent steps for legal staircase")

    # General finance → OpenAI
    if is_general_finance(query):
        return chat(query), "No agent action required."

    return chat(query), chat("Give compliant agent suggestions.")

# ======================================================
# QUERY INPUT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID")

if query:
    answer, tips = answer_query(query)

    st.markdown("""
    <div class="card highlight">
    <h3>🧠 System Response</h3>
    <p>{}</p>
    </div>
    """.format(answer), unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h4>🎧 Agent Suggestions</h4>
    <div class="action">{}</div>
    </div>
    """.format(tips), unsafe_allow_html=True)

    # ==================================================
    # PDF DOWNLOAD
    # ==================================================
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(40, 800, "NBFC Legal Intelligence Report")
    pdf.drawString(40, 780, f"Query: {query}")
    pdf.drawString(40, 760, f"Answer: {answer}")
    pdf.drawString(40, 720, f"Agent Suggestions: {tips}")
    pdf.save()
    buffer.seek(0)

    st.download_button(
        "📄 Download Summary (PDF)",
        data=buffer,
        file_name="NBFC_Legal_Response.pdf",
        mime="application/pdf"
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p class="small" style="text-align:center;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence
</p>
""", unsafe_allow_html=True)
