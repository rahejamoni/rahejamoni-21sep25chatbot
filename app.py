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
# GLOBAL CSS (CLEAN ENTERPRISE UI)
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
}

.hero {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub {
    color: #94a3b8;
    font-size: 1.1rem;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 600;
}

.small {
    color: #94a3b8;
    font-size: 0.9rem;
}

.response-box {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
}

.agent-box {
    background: rgba(45,212,191,0.08);
    border-left: 4px solid #2dd4bf;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR (MINIMAL)
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("### Intel Center")
    st.caption("NBFC Legal Decision Support")
    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

# ======================================================
# HERO SECTION
# ======================================================
st.markdown('<div class="hero">Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">Advanced decision-support for NBFC collections and legal risk mitigation.</p>', unsafe_allow_html=True)

# ======================================================
# INTRODUCTION + USAGE
# ======================================================
intro_col, usage_col = st.columns(2)

with intro_col:
    st.markdown("""
    <div class="card">
    <div class="feature-title">🔍 What does this assistant do?</div>
    <ul class="small">
        <li>Explains NBFC legal notices and recovery stages</li>
        <li>Interprets SARFAESI, Section 138 & arbitration steps</li>
        <li>Fetches LAN-level recovery status</li>
        <li>Suggests compliant customer communication</li>
    </ul>
    ⚠️ For operational guidance only. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

with usage_col:
    st.markdown("""
    <div class="card">
    <div class="feature-title">ℹ️ How to use</div>
    <ul class="small">
        <li>Ask a legal or collections question</li>
        <li>Enter a LAN ID (e.g. 22222)</li>
        <li>Review system response and agent suggestions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# CORE CAPABILITIES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <div class="feature-title">⚖️ Legal Staircase</div>
    <p class="small">Step-by-step interpretation of legal recovery processes and timelines.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <div class="feature-title">🔍 LAN Intelligence</div>
    <p class="small">Loan status, notice history, and recovery stage using LAN.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <div class="feature-title">📞 Communication</div>
    <p class="small">Polite, compliant call scripts tailored to case severity.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ASK THE ASSISTANT
# ======================================================
st.markdown("## 💬 Ask the Legal Assistant")

query = st.text_input(
    "",
    placeholder="e.g. What is a pre-sale notice? | Enter LAN ID",
    label_visibility="collapsed"
)

# ======================================================
# RESPONSE SECTION
# ======================================================
if query:
    # 🔁 Replace this with your actual answer_query() logic
    answer = (
        "The staircase of consumer finance refers to progressive recovery steps "
        "starting from reminder communication, followed by legal notices, "
        "repossession, and auction if dues remain unpaid."
    )

    agent_tips = [
        "Confirm customer awareness of the current recovery stage.",
        "Clearly explain consequences while maintaining a polite tone.",
        "Offer immediate resolution options if payment intent exists."
    ]

    st.markdown("""
    <div class="response-box">
    <strong>🧠 System Response</strong><br><br>
    """ + answer + """
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="agent-box">
    <strong>🎧 Agent Suggestions</strong>
    <ul>
    """ + "".join([f"<li>{t}</li>" for t in agent_tips]) + """
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#94a3b8; font-size:14px;">
Created by <b>Mohit Raheja</b> | NBFC Legal Intelligence Assistant
</p>
""", unsafe_allow_html=True)
