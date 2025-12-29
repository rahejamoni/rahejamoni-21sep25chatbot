import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import io

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# OPENAI CLIENT
# ======================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# LOAD EXCEL (RAG SOURCE)
# ======================================================
@st.cache_data
def load_legal_data():
    df = pd.read_excel("legal_staircase.xlsx")
    df = df.fillna("")
    df["combined"] = (
        df.astype(str)
        .apply(lambda x: " | ".join(x), axis=1)
        .str.lower()
    )
    return df

legal_df = load_legal_data()

# ======================================================
# RAG RETRIEVAL (KEYWORD MATCH)
# ======================================================
def retrieve_from_excel(query, df, top_k=3):
    query = query.lower()
    scores = []

    for idx, row in df.iterrows():
        score = sum(1 for w in query.split() if w in row["combined"])
        scores.append(score)

    df["score"] = scores
    results = df.sort_values("score", ascending=False).head(top_k)
    results = results[results["score"] > 0]

    if results.empty:
        return None

    context = "\n".join(results["combined"].tolist())
    return context

# ======================================================
# OPENAI FALLBACK
# ======================================================
def ask_openai(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an NBFC legal intelligence assistant. "
                    "Answer clearly and concisely. If it is general knowledge "
                    "(e.g. capital of India), answer directly."
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ======================================================
# AGENT ADVICE GENERATOR
# ======================================================
def agent_advice(answer):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior NBFC compliance manager. "
                    "Give clear, actionable advice for collection agents. "
                    "Ensure RBI, SARFAESI, and consumer protection compliance."
                )
            },
            {"role": "user", "content": answer}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ======================================================
# UI STYLES
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #0f172a, #020617); color:#f8fafc; }
.hero { font-size:2.6rem; font-weight:800; color:#60a5fa; }
.sub { color:#94a3b8; margin-bottom:1.5rem; }
.card {
    background: rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px;
    padding:20px;
}
.answer {
    border-left:4px solid #3b82f6;
    padding:18px;
    background: rgba(15,23,42,0.8);
    border-radius:10px;
}
.advice {
    border-left:4px solid #22c55e;
    padding:18px;
    background: rgba(15,23,42,0.8);
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================
st.markdown('<div class="hero">NBFC Legal Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">AI-powered decision support for NBFC collections & legal compliance</div>',
    unsafe_allow_html=True
)

# ======================================================
# INTRO SECTIONS
# ======================================================
colA, colB = st.columns([2, 1])

with colA:
    st.markdown("""
    <div class="card">
    <h4>🔍 What does this assistant do?</h4>
    <ul>
        <li>Explains NBFC legal notices & recovery stages</li>
        <li>Interprets SARFAESI, Section 138 & arbitration</li>
        <li>Fetches LAN-level recovery context</li>
        <li>Suggests compliant customer communication</li>
    </ul>
    ⚠️ For operational guidance only. Not legal advice.
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
    <div class="card">
    <h4>ℹ️ How to use</h4>
    <ul>
        <li>Ask a legal or collections question</li>
        <li>Enter a LAN ID (e.g. 22222)</li>
        <li>Review system answer + agent advice</li>
        <li>Download response if required</li>
    </ul>
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
    Recovery flow from reminder → SARFAESI → litigation
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h4>🔍 LAN Intelligence</h4>
    Loan-level recovery & notice context
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h4>📞 Communication</h4>
    RBI-compliant scripts & agent guidance
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# QUERY INPUT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID...")

if query:
    with st.spinner("Analyzing..."):
        context = retrieve_from_excel(query, legal_df)

        if context:
            system_answer = ask_openai(
                f"Answer using ONLY this context:\n{context}\n\nQuestion: {query}"
            )
        else:
            system_answer = ask_openai(query)

        advice = agent_advice(system_answer)

    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 💡 Legal / System Answer")
        st.markdown(f'<div class="answer">{system_answer}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### 🎧 Agent Advice")
        st.markdown(f'<div class="advice">{advice}</div>', unsafe_allow_html=True)

    # ==================================================
    # DOWNLOAD FEATURE
    # ==================================================
    download_text = f"""
QUERY:
{query}

SYSTEM ANSWER:
{system_answer}

AGENT ADVICE:
{advice}
"""
    st.download_button(
        "📄 Download Response",
        data=download_text,
        file_name="nbfc_legal_response.txt",
        mime="text/plain"
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr style="opacity:0.2">
<div style="text-align:center; color:#64748b; font-size:0.8rem;">
Designed by <strong>Mohit Raheja</strong> | NBFC Legal Intelligence System
</div>
""", unsafe_allow_html=True)
