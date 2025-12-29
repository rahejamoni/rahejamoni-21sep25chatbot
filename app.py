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
# CSS – COMPACT, APP-LIKE UI
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #e5e7eb;
}
.hero-text {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-text {
    color: #94a3b8;
    margin-bottom: 18px;
}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
}
.small {
    font-size: 0.85rem;
    color: #9ca3af;
}
.answer-card {
    background: rgba(2,6,23,0.8);
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 18px;
}
.agent-card {
    background: rgba(16,185,129,0.08);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 16px;
}
.step {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="hero-text">NBFC Legal & Collections Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI decision-support system for compliant NBFC recovery operations</div>', unsafe_allow_html=True)

# ======================================================
# QUICK INFO ROW (NO DUPLICATION)
# ======================================================
info_l, info_r = st.columns([2, 1])

with info_l:
    st.markdown("""
    <div class="card">
    <b>🔍 What does this assistant do?</b>
    <ul class="small">
        <li>Explains NBFC legal notices & recovery staircase</li>
        <li>Fetches LAN-based recovery status</li>
        <li>Provides compliant calling guidance</li>
    </ul>
    <span class="small">⚠️ Operational guidance only. Not legal advice.</span>
    </div>
    """, unsafe_allow_html=True)

with info_r:
    st.markdown("""
    <div class="card">
    <b>⚡ Quick guide</b>
    <ul class="small">
        <li>Ask legal / collections questions</li>
        <li>Enter LAN ID (e.g. 22222)</li>
        <li>Review agent action steps</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# CHAT INPUT (VISIBLE IMMEDIATELY)
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID…")

# ======================================================
# MOCK LOGIC (REPLACE WITH YOUR REAL RAG / LAN LOGIC)
# ======================================================
def get_system_answer(q):
    # Replace this with your existing RAG + LAN logic
    return (
        "The staircase of consumer finance refers to progressive recovery stages "
        "starting from soft reminders, legal notices, asset possession, and finally auction."
    )

def get_agent_suggestions(q):
    # Replace with your LLM-based agent strategy function
    return [
        "Verify the current DPD and last payment date before initiating discussion.",
        "Clearly explain the next legal step while maintaining a calm and respectful tone.",
        "Offer a short settlement window to prevent escalation."
    ]

# ======================================================
# RESPONSE SECTION (ANSWER + AGENT SUGGESTIONS SIDE BY SIDE)
# ======================================================
if query:
    answer = get_system_answer(query)
    agent_steps = get_agent_suggestions(query)

    ans_col, agent_col = st.columns([2, 1])

    with ans_col:
        st.markdown("""
        <div class="answer-card">
        <b>🧠 System Interpretation</b>
        <p style="margin-top:10px; line-height:1.55; font-size:0.95rem;">
        """ + answer + """
        </p>
        </div>
        """, unsafe_allow_html=True)

    with agent_col:
        st.markdown("""
        <div class="agent-card">
        <b>🎧 Agent Suggestions</b>
        """, unsafe_allow_html=True)

        for i, step in enumerate(agent_steps, 1):
            st.markdown(f"""
            <div class="step">
            <b>Step {i}:</b> {step}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# OPTIONAL PDF EXPORT (KEEP OR REMOVE)
# ======================================================
def generate_pdf(query, answer, steps):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 50, "NBFC LEGAL INTELLIGENCE SUMMARY")

    p.setFont("Helvetica", 10)
    p.drawString(50, height - 70, f"Query: {query}")

    text = p.beginText(50, height - 100)
    text.setLeading(14)
    text.textLine("System Interpretation:")
    text.textLine(answer)
    text.textLine("")
    text.textLine("Agent Suggestions:")
    for s in steps:
        text.textLine(f"- {s}")

    p.drawText(text)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

if query:
    pdf = generate_pdf(query, answer, agent_steps)
    st.download_button(
        "📄 Download Summary (PDF)",
        pdf,
        file_name="NBFC_Intelligence_Report.pdf",
        mime="application/pdf"
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; font-size:0.8rem; color:#9ca3af;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence
</p>
""", unsafe_allow_html=True)
