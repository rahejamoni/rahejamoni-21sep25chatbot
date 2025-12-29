import os
import re
import io
import pickle
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
    page_title="NBFC Intel | Decision Hub",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# ADVANCED CSS - MIDNIGHT GOLD THEME
# ======================================================
st.markdown("""
<style>
/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #040911 0%, #0a192f 100%);
    color: #e6f1ff;
}

/* Typography */
.hero-text {
    font-size: 3rem;
    font-weight: 850;
    background: linear-gradient(90deg, #e1b12c, #fbc531);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.sub-text {
    color: #8892b0;
    font-size: 1.1rem;
    margin-bottom: 30px;
    font-weight: 300;
}

/* Bento Cards */
.card {
    background: rgba(17, 34, 64, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100, 255, 218, 0.1);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
}
.card:hover {
    border: 1px solid rgba(225, 177, 44, 0.4);
    transform: translateY(-2px);
}

/* Interpretation & Suggestions */
.answer-card {
    background: rgba(10, 25, 47, 0.9);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid #112240;
    box-shadow: 0 10px 30px -15px rgba(2, 12, 27, 0.7);
}

.agent-header {
    color: #fbc531;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
}

.step-box {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
    border-left: 3px solid #e1b12c;
}

/* Download Button Styling */
.stDownloadButton button {
    background-color: transparent !important;
    color: #fbc531 !important;
    border: 1px solid #fbc531 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    transition: 0.3s !important;
}
.stDownloadButton button:hover {
    background-color: rgba(225, 177, 44, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER SECTION
# ======================================================
st.markdown('<div class="hero-text">Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Strategic Decision-Support for NBFC Collections & Compliance</div>', unsafe_allow_html=True)

# Dashboard Feature Icons
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown('<div class="card"><b>⚖️ Regulatory Guard</b><br><span style="font-size:0.8rem; color:#8892b0;">Ensures all actions align with RBI recovery guidelines.</span></div>', unsafe_allow_html=True)
with col_f2:
    st.markdown('<div class="card"><b>🔍 Direct Retrieval</b><br><span style="font-size:0.8rem; color:#8892b0;">Real-time LAN status mapping and legal history.</span></div>', unsafe_allow_html=True)
with col_f3:
    st.markdown('<div class="card"><b>🗣️ Script Intelligence</b><br><span style="font-size:0.8rem; color:#8892b0;">Dynamic agent scripts for high-conversion negotiations.</span></div>', unsafe_allow_html=True)

st.write(" ")

# ======================================================
# CORE LOGIC (MOCK)
# ======================================================
def get_system_answer(q):
    # Place your actual RAG/Excel logic here
    return (
        "Under Section 138 of the NI Act, a demand notice must be served within 30 days of "
        "cheque dishonor. The borrower then has 15 days to settle the payment before "
        "a criminal complaint can be filed in the jurisdictional court."
    )

def get_agent_suggestions(q):
    return [
        "Confirm if the customer received the physical notice via Speed Post tracking.",
        "Inform the customer that a 'Civil Settlement' is still possible before court filing.",
        "Document the customer's refusal or promise-to-pay (PTP) in the CRM immediately."
    ]

# ======================================================
# INTERACTION ZONE
# ======================================================
query = st.chat_input("Enter LAN ID or ask a recovery question...")

if query:
    answer = get_system_answer(query)
    agent_steps = get_agent_suggestions(query)

    # Main Grid for Results
    res_col, side_col = st.columns([2, 1])

    with res_col:
        st.markdown(f"""
        <div class="answer-card">
            <h4 style="color:#ccd6f6; margin-top:0;">💡 Intelligence Output</h4>
            <p style="color:#8892b0; font-size:1.05rem; line-height:1.7;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)

    with side_col:
        st.markdown('<div class="agent-header">⚡ Agent Action Plan</div>', unsafe_allow_html=True)
        for i, step in enumerate(agent_steps, 1):
            st.markdown(f"""
            <div class="step-box">
                <span style="color:#fbc531; font-weight:bold; margin-right:8px;">0{i}</span>
                <span style="font-size:0.9rem;">{step}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # PDF Generation & Download
        def generate_pdf(q, a, s):
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica-Bold", 16)
            p.drawString(50, 750, "NBFC LEGAL ADVISORY REPORT")
            p.setFont("Helvetica", 12)
            p.drawString(50, 730, f"Query: {q}")
            p.line(50, 720, 550, 720)
            p.showPage()
            p.save()
            buffer.seek(0)
            return buffer

        pdf = generate_pdf(query, answer, agent_steps)
        st.download_button("📄 Export to PDF", pdf, file_name="NBFC_Advisor.pdf")

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div style="margin-top: 100px; text-align:center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
    <p style="color: #495670; font-size: 0.85rem;">
        <b>Mohit Raheja</b> | Applied AI Division | Secure Enterprise Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
