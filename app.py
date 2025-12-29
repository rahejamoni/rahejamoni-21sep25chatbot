import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# GLOBAL CSS (Clean Enterprise UI)
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}

.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
}

.hero {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub {
    color: #94a3b8;
    font-size: 1.05rem;
}

.tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 14px;
    font-size: 12px;
    background: rgba(45,212,191,0.12);
    color: #2dd4bf;
    border: 1px solid rgba(45,212,191,0.25);
}

.answer-box {
    background: rgba(15,23,42,0.75);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px;
}

.tip-box {
    background: rgba(45,212,191,0.08);
    border-left: 4px solid #2dd4bf;
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR (INTEL CENTER ONLY)
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("## Intel Center")
    st.markdown("---")

    st.markdown("""
    **What does this assistant do?**
    - Explains NBFC legal notices & recovery stages  
    - Interprets SARFAESI, Section 138 & arbitration  
    - Fetches LAN-level recovery status  
    - Suggests compliant customer communication  

    ⚠️ *For operational guidance only. Not legal advice.*
    """)

    st.markdown("---")

    st.markdown("""
    **ℹ️ How to use**
    - Ask a legal or collections question  
    - Enter a LAN ID (e.g. 22222)  
    - Review system response & agent suggestions  
    """)

    st.markdown("---")
    st.caption("Designed by **Mohit Raheja**")

# ======================================================
# MAIN HEADER
# ======================================================
st.markdown('<div class="hero">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Advanced decision-support for NBFC collections and legal risk mitigation</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# FEATURE BOXES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <h4 style="color:#3b82f6;">⚖️ Legal Staircase</h4>
        <p class="sub">Step-by-step interpretation of SARFAESI, Section 138, and arbitration processes.</p>
        <span class="tag">Updated: RBI 2024</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <h4 style="color:#2dd4bf;">🔍 LAN Intelligence</h4>
        <p class="sub">Real-time loan status, notice history, and recovery stage using LAN.</p>
        <span class="tag">Active Database</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <h4 style="color:#f59e0b;">📞 Communication</h4>
        <p class="sub">Compliant call scripts and negotiation guidance for agents.</p>
        <span class="tag">Audit-Ready</span>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# QUERY INPUT
# ======================================================
st.markdown("<br>", unsafe_allow_html=True)
query = st.chat_input("Ask a legal question or enter LAN ID (e.g. 22222)")

# ======================================================
# MOCK LOGIC (REPLACE WITH YOUR REAL answer_query)
# ======================================================
if query:
    # Example outputs (plug your real logic here)
    answer = (
        "The staircase of consumer finance refers to progressive recovery stages followed by NBFCs — "
        "starting from reminders, followed by demand notice, SARFAESI actions, and finally auction or legal recovery."
    )

    agent_tips = [
        "Confirm whether the customer has received the last notice.",
        "Explain consequences calmly and compliantly.",
        "Ask for a realistic payment commitment date."
    ]

    col_ans, col_tip = st.columns([2.2, 1.3])

    with col_ans:
        st.markdown("### 🧠 System Response")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    with col_tip:
        st.markdown("### 🎧 Agent Suggestions")
        for t in agent_tips:
            st.markdown(f'<div class="tip-box">• {t}</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#64748b; font-size:13px;">
NBFC Legal Intelligence Assistant | Created by <b>Mohit Raheja</b>
</p>
""", unsafe_allow_html=True)
