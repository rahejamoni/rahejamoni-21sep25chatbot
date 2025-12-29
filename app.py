import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="NBFC Intel | Legal & Collections",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# CUSTOM CSS – ENTERPRISE UI
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e293b, #0f172a);
    color: #f8fafc;
}
.hero-text {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}
.bento-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
}
.status-tag {
    display:inline-block;
    padding:4px 12px;
    border-radius:20px;
    font-size:12px;
    background:rgba(45,212,191,0.15);
    color:#2dd4bf;
}
.response-area {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 18px;
}
.action-box {
    background: rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(59,130,246,0.25);
}
.stChatInputContainer {
    position: sticky;
    bottom: 0;
    background: #0f172a;
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG (stable)
# ======================================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("## Intel Center")
    st.markdown("### 📊 Live Analytics")
    st.metric("Compliance Score", "98.2%", "+0.4%")
    st.metric("Case Resolution", "1.4k", "+12%")
    st.caption("v2.4.0 High-Performance Edition")
    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

# ======================================================
# HERO SECTION
# ======================================================
st.markdown('<h1 class="hero-text">Legal Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Advanced decision-support for NBFC collections and legal risk mitigation.</p>', unsafe_allow_html=True)

# ======================================================
# FEATURE CARDS
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#3b82f6;">⚖️ Legal Staircase</h4>
        <p class="sub-text">Interpret SARFAESI, Section 138 & Arbitration steps.</p>
        <span class="status-tag">RBI 2024</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#2dd4bf;">🔍 LAN Intelligence</h4>
        <p class="sub-text">Fetch notice history & recovery status via LAN.</p>
        <span class="status-tag">Live Data</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#f59e0b;">📞 Communication</h4>
        <p class="sub-text">Get compliant, audit-ready call scripts.</p>
        <span class="status-tag">Agent Ready</span>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# QUERY INPUT (STICKY, CHAT STYLE)
# ======================================================
st.markdown("### 🧬 Query Intelligence")
query = st.chat_input("Ask a legal question or enter LAN ID (e.g. 22222)")

# ======================================================
# ANSWER ENGINE (DEMO + AGENT SUGGESTIONS)
# ======================================================
if query:
    with st.spinner("Processing through Legal Intelligence Engine..."):

        # ---- DEMO LOGIC (replace with your real RAG later) ----
        is_lan = re.search(r"\b\d{3,}\b", query)

        if is_lan:
            answer = (
                f"LAN **{is_lan.group()}** is currently under **Pre-Sale Notice stage**. "
                "The notice was sent and the cooling period is active."
            )
            agent_steps = [
                "Verify notice delivery acknowledgment.",
                "Check if any payment was made post-notice.",
                "Offer last-mile settlement before auction."
            ]
        else:
            answer = (
                "The staircase of consumer finance refers to progressive legal and financial "
                "actions taken as a loan moves from early delinquency to recovery, including "
                "reminders, notices, possession, and auction."
            )
            agent_steps = [
                "Confirm delinquency stage (DPD bucket).",
                "Explain next legal step clearly to the customer.",
                "Seek a payment commitment date."
            ]

    # ======================================================
    # COMPACT RESPONSE LAYOUT (NO SCROLL)
    # ======================================================
    res_col, act_col = st.columns([2.2, 1])

    with res_col:
        st.markdown("#### 💡 Legal Interpretation")
        st.markdown(f"""
        <div class="response-area">
            {answer}
        </div>
        """, unsafe_allow_html=True)

    with act_col:
        st.markdown("#### 🚀 Agent Suggestions")
        for step in agent_steps:
            st.markdown(f"""
            <div class="action-box">• {step}</div>
            """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#64748b; font-size:0.85rem;">
Designed by <strong>Mohit Raheja</strong> | Applied AI & Decision Intelligence<br>
Enterprise-grade NBFC Legal Assistant
</p>
""", unsafe_allow_html=True)
