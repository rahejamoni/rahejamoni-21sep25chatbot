import os
import re
import pickle
import io
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# CUSTOM CSS (ENTERPRISE UI)
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.9);
    border-right: 1px solid rgba(255,255,255,0.1);
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
    font-size: 1.1rem;
}
.bento-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 22px;
}
.response-card {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    padding: 20px;
    border-radius: 12px;
}
.agent-card {
    background: rgba(45,212,191,0.08);
    border-left: 4px solid #2dd4bf;
    padding: 18px;
    border-radius: 12px;
}
.small-text {
    color:#94a3b8;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (ASSUMES SECRET SET)
# ======================================================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# PDF GENERATOR
# ======================================================
def generate_pdf(query, answer, tips):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "NBFC LEGAL INTELLIGENCE REPORT")

    p.setFont("Helvetica", 10)
    p.drawString(50, height - 70, f"Generated on: {datetime.now()}")

    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 110, "Query:")
    p.setFont("Helvetica", 11)
    p.drawString(50, height - 130, query)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 170, "System Interpretation:")
    text = p.beginText(50, height - 190)
    text.setFont("Helvetica", 11)
    for line in answer.split("\n"):
        text.textLine(line)
    p.drawText(text)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, text.getY() - 30, "Agent Guidance:")
    p.setFont("Helvetica", 11)
    p.drawString(50, text.getY() - 50, tips)

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ======================================================
# SIDEBAR (INTEL CENTER)
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("## Intel Center")
    st.markdown("---")

    st.markdown("### 🔍 What does this assistant do?")
    st.markdown("""
- Explains NBFC legal notices and recovery stages  
- Interprets SARFAESI, Section 138 & arbitration steps  
- Fetches LAN-level recovery status  
- Suggests compliant customer communication  

⚠️ *For operational guidance only. Not legal advice.*
""")

    st.markdown("---")

    st.markdown("### ℹ️ How to use")
    st.markdown("""
- Ask a legal or collections question  
- Enter a LAN ID (e.g. 22222)  
- Review system response and agent suggestions  
""")

    st.markdown("---")
    st.caption("Designed by **Mohit Raheja**")

# ======================================================
# MAIN HEADER
# ======================================================
st.markdown('<div class="hero-text">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered decision support for NBFC collections & legal compliance</p>', unsafe_allow_html=True)

# ======================================================
# THREE CORE MODULES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="bento-card">
    <h4>⚖️ Legal Staircase</h4>
    <p class="small-text">
    Step-by-step interpretation of SARFAESI, Section 138, Demand, Pre-Sale & Arbitration.
    </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="bento-card">
    <h4>🔍 LAN Intelligence</h4>
    <p class="small-text">
    Fetch recovery status, notice stage and action readiness using LAN ID.
    </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="bento-card">
    <h4>📞 Communication</h4>
    <p class="small-text">
    Polite, compliant and audit-safe calling guidance for agents.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# QUERY INPUT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID...")

if query:
    with st.spinner("Analyzing through Legal Intelligence Engine..."):
        # ---- MOCK LOGIC (replace with your real logic) ----
        answer_text = (
            "A pre-sale notice is a mandatory legal communication issued before auction "
            "informing the borrower of outstanding dues and proposed sale date."
        )

        agent_tips = (
            "1. Confirm receipt of the notice.\n"
            "2. Inform borrower of limited time before auction.\n"
            "3. Encourage immediate payment or settlement discussion."
        )

    col_ans, col_tip = st.columns([2, 1])

    with col_ans:
        st.markdown("### 🧠 System Response")
        st.markdown(f"""
        <div class="response-card">
        {answer_text}
        </div>
        """, unsafe_allow_html=True)

    with col_tip:
        st.markdown("### 🎧 Agent Suggestions")
        st.markdown(f"""
        <div class="agent-card">
        {agent_tips}
        </div>
        """, unsafe_allow_html=True)

    # PDF DOWNLOAD
    pdf = generate_pdf(query, answer_text, agent_tips)
    st.download_button(
        label="📄 Download Legal Summary (PDF)",
        data=pdf,
        file_name="NBFC_Legal_Summary.pdf",
        mime="application/pdf",
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#64748b; font-size:13px;">
NBFC Legal Intelligence Hub | Applied AI Project<br>
Created by <b>Mohit Raheja</b>
</p>
""", unsafe_allow_html=True)
