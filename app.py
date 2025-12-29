import os
import re
import pickle
import io
import textwrap
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
# OPENAI CONFIG
# ======================================================
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please set the OPENAI_API_KEY in Streamlit Secrets.")

# ======================================================
# PDF GENERATOR (WITH LINE WRAPPING)
# ======================================================
def generate_pdf(query, answer, tips):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    line_width = 90  # Characters per line

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(margin, height - 50, "NBFC LEGAL INTELLIGENCE REPORT")
    
    p.setFont("Helvetica", 10)
    p.drawString(margin, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p.line(margin, height - 80, width - margin, height - 80)

    # User Query Section
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, height - 110, "User Query:")
    p.setFont("Helvetica", 11)
    wrapped_query = textwrap.wrap(query, width=line_width)
    y = height - 130
    for line in wrapped_query:
        p.drawString(margin, y, line)
        y -= 15

    # System Interpretation Section
    y -= 20
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, y, "System Interpretation:")
    p.setFont("Helvetica", 11)
    y -= 20
    
    # Wrap and print the answer text
    wrapped_answer = textwrap.wrap(answer, width=line_width)
    for line in wrapped_answer:
        if y < 50: # Simple page break check
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 11)
        p.drawString(margin, y, line)
        y -= 15

    # Agent Guidance Section
    y -= 25
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, y, "Agent Strategic Guidance:")
    p.setFont("Helvetica-Oblique", 11)
    y -= 20
    
    wrapped_tips = textwrap.wrap(tips, width=line_width)
    for line in wrapped_tips:
        if y < 50:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica-Oblique", 11)
        p.drawString(margin, y, line)
        y -= 15

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## Intel Center")
    st.markdown("---")
    st.markdown("### 🔍 Capabilities")
    st.markdown("""
    - Legal Staircase Interpretation
    - LAN-level Status Retrieval
    - Compliance Check
    - Agent Call Strategy
    """)
    st.markdown("---")
    st.caption("Designed by **Mohit Raheja**")

# ======================================================
# MAIN HEADER
# ======================================================
st.markdown('<div class="hero-text">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered decision support for NBFC collections & legal compliance</p>', unsafe_allow_html=True)

# Bento Modules
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="bento-card"><h4>⚖️ Legal Staircase</h4><p class="small-text">SARFAESI, Sec 138, and Arbitration analysis.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="bento-card"><h4>🔍 LAN Intelligence</h4><p class="small-text">Real-time status tracking via Loan Account Number.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="bento-card"><h4>📞 Communication</h4><p class="small-text">Audit-safe calling scripts and guidance.</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# QUERY ENGINE
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID...")

if query:
    with st.spinner("Analyzing..."):
        # Logic Placeholder: Replace these with your actual model/dataframe calls
        answer_text = "A pre-sale notice is a mandatory legal communication issued before auction informing the borrower of outstanding dues and proposed sale date. Under current NBFC guidelines, this serves as the final opportunity for settlement before the asset is liquidated."
        agent_tips = "1. Confirm receipt of the notice. 2. Inform borrower of limited time before auction. 3. Encourage immediate payment or a structured settlement discussion to prevent the auction process."

    col_ans, col_tip = st.columns([2, 1])

    with col_ans:
        st.markdown("### 🧠 System Response")
        st.markdown(f'<div class="response-card">{answer_text}</div>', unsafe_allow_html=True)

    with col_tip:
        st.markdown("### 🎧 Agent Suggestions")
        st.markdown(f'<div class="agent-card">{agent_tips}</div>', unsafe_allow_html=True)

    # PDF Download Button
    st.markdown("---")
    pdf_output = generate_pdf(query, answer_text, agent_tips)
    st.download_button(
        label="📄 Download Legal Summary (PDF)",
        data=pdf_output,
        file_name=f"NBFC_Report_{datetime.now().strftime('%d%m%Y')}.pdf",
        mime="application/pdf"
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#64748b; font-size:13px;">
NBFC Legal Intelligence Hub | Created by <b>Mohit Raheja</b>
</p>
""", unsafe_allow_html=True)
