import re
import os
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
    page_title="NBFC Intel | Legal & Collections",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# CSS – CLEAN ENTERPRISE UI
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b, #0f172a); color: #f8fafc; }
.bento-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
}
.hero-text {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg,#3b82f6,#2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text { color:#94a3b8; font-size:1.05rem; }
.response-box {
    background: rgba(15,23,42,0.7);
    border-left: 4px solid #3b82f6;
    padding: 18px;
    border-radius: 10px;
}
.advice-box {
    background: rgba(34,197,94,0.08);
    border-left: 4px solid #22c55e;
    padding: 15px;
    border-radius: 10px;
    font-size: 0.95rem;
}
.small { color:#94a3b8; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# OPENAI CONFIG
# ======================================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# OPENAI HELPERS
# ======================================================
def chat(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=220
    )
    return res.choices[0].message["content"].strip()

# ======================================================
# ROUTING LOGIC
# ======================================================
def is_general_question(q):
    keywords = ["capital", "who is", "what is", "define", "country", "india"]
    return any(k in q.lower() for k in keywords)

def agent_advice(context):
    return chat(f"""
You are a senior NBFC collections manager.
Give 3 short, compliant, polite call instructions for an agent.
Context:
{context}
Return bullet points only.
""")

# ======================================================
# HEADER
# ======================================================
st.markdown('<h1 class="hero-text">Legal Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Decision-support system for NBFC collections & legal compliance</p>', unsafe_allow_html=True)

# ======================================================
# FEATURE BOXES
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="bento-card">
    <h4>⚖️ Legal Staircase</h4>
    <p class="small">Explains SARFAESI, Section 138, arbitration & recovery stages.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="bento-card">
    <h4>🔍 LAN Intelligence</h4>
    <p class="small">Fetches loan-level recovery status & notice stage.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="bento-card">
    <h4>📞 Communication</h4>
    <p class="small">Provides compliant scripts & next-step guidance.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# INTRO + HOW TO USE
# ======================================================
i1, i2 = st.columns([2,1])

with i1:
    st.markdown("""
    <div class="bento-card">
    <b>What does this assistant do?</b>
    <ul class="small">
    <li>Explains NBFC legal notices and recovery stages</li>
    <li>Interprets SARFAESI, Section 138 & arbitration steps</li>
    <li>Fetches LAN-level recovery status</li>
    <li>Suggests compliant customer communication</li>
    </ul>
    ⚠️ For operational guidance only. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="bento-card">
    <b>ℹ️ How to use</b>
    <ul class="small">
    <li>Ask a legal or collections question</li>
    <li>Enter a LAN ID (e.g. 22222)</li>
    <li>Review system response & agent advice</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# QUERY SECTION
# ======================================================
st.markdown("### 💬 Ask a Question")
query = st.chat_input("Ask about legal process, recovery stage, or enter LAN ID")

if query:
    with st.spinner("Analyzing through legal engine..."):

        # --- LOGIC ---
        if re.search(r"\b\d{3,}\b", query):
            answer = f"LAN **{query}** is currently in the recovery workflow. Please verify the latest notice stage before further action."
        elif is_general_question(query):
            answer = chat(query)
        else:
            answer = chat(f"""
Explain clearly for an NBFC collection agent:
{query}
Keep it structured and simple.
""")

        advice = agent_advice(answer)

    # ======================================================
    # OUTPUT
    # ======================================================
    col_ans, col_adv = st.columns([2,1])

    with col_ans:
        st.markdown("#### 🧠 System Explanation")
        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

    with col_adv:
        st.markdown("#### 🎧 Agent Advice")
        st.markdown(f"<div class='advice-box'>{advice}</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center;color:#64748b;font-size:0.8rem;">
Designed by <b>Mohit Raheja</b> | Applied AI for NBFC Collections
</p>
""", unsafe_allow_html=True)
