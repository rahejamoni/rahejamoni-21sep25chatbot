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
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="NBFC Intel | Decision Hub",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# CSS: GLASSMORPHISM & PRO UI
# ======================================================
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at top left, #0f172a, #1e293b); color: #f8fafc; }
    .hero-text { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #3b82f6, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; }
    .bento-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 20px; transition: 0.3s; }
    .bento-card:hover { border-color: #3b82f6; background: rgba(255, 255, 255, 0.05); }
    .response-card { background: rgba(15, 23, 42, 0.8); border-radius: 12px; padding: 25px; border-left: 5px solid #3b82f6; margin-top: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); }
    .action-step { background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(59, 130, 246, 0.2); }
</style>
""", unsafe_allow_html=True)

# ======================================================
# PDF GENERATOR FUNCTION
# ======================================================
def generate_pdf(query, answer, tips):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.setStrokeColor(colors.dodgerblue)
    p.drawString(50, height - 50, "NBFC LEGAL INTELLIGENCE REPORT")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 65, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p.line(50, height - 75, width - 50, height - 75)

    # Content
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 100, "Query Analysis:")
    p.setFont("Helvetica", 11)
    
    # Text wrapping for Answer
    text_object = p.beginText(50, height - 120)
    text_object.setFont("Helvetica", 11)
    text_object.setLeading(14)
    lines = answer.split('\n')
    for line in lines:
        text_object.textLine(line)
    p.drawText(text_object)

    # Agent Guidance
    p.setFont("Helvetica-Bold", 12)
    curr_y = text_object.getY() - 30
    p.drawString(50, curr_y, "Recommended Agent Action:")
    p.setFont("Helvetica-Oblique", 11)
    p.drawString(50, curr_y - 20, tips[:100] + "...") # Simplified for demo

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ======================================================
# SIDEBAR & STATS
# ======================================================
with st.sidebar:
    st.markdown("### 🛠️ Intelligence Tools")
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Metrics")
    st.progress(85, text="Daily Recovery Target")
    st.caption("Current Accuracy: 99.4%")

# ======================================================
# MAIN INTERFACE
# ======================================================
st.markdown('<h1 class="hero-text">Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8; margin-bottom: 30px;">Automated Decision Support for NBFC Collections</p>', unsafe_allow_html=True)

# Feature Explanation (Bento)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="bento-card"><h4>⚖️ Compliance</h4><p style="font-size:0.85rem; color:#94a3b8;">Instant legal staircase interpretations for SARFAESI, Sec 138, and Arbitration.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="bento-card"><h4>🔍 Retrieval</h4><p style="font-size:0.85rem; color:#94a3b8;">Enter LAN IDs to fetch notice history and recovery status from internal databases.</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="bento-card"><h4>📞 Strategy</h4><p style="font-size:0.85rem; color:#94a3b8;">Receive compliant scripts and negotiation tactics tailored to the customer profile.</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# CHAT ENGINE
# ======================================================
query = st.chat_input("Enter LAN ID or Legal Question...")

if query:
    # --- PROCESSSING LOGIC (Call your existing functions here) ---
    # Example mock results:
    ans_text = "A pre-sale notice is a legal mandatory communication sent 15 days prior to auction. It must include the outstanding balance and the specific date of the proposed sale. Failure to send this can invalidate the recovery process."
    tips_text = "Explain the urgency: 'Mr. Customer, this is the final notice before your asset is moved to auction. We can still halt this if payment is made today.'"

    # 1. Display Intelligence
    st.markdown(f"""
    <div class="response-card">
        <h3 style="margin-top:0; color:#3b82f6;">💡 Legal Interpretation</h3>
        <p style="line-height:1.6; font-size:1.1rem;">{ans_text}</p>
        <hr style="opacity:0.1">
        <h4 style="color:#2dd4bf;">🚀 Agent Action Plan</h4>
        <div class="action-step"><strong>Primary:</strong> Verify Notice Receipt Date.</div>
        <div class="action-step"><strong>Secondary:</strong> Initiate "Final Settlement" call script.</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. PDF Download Section
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_file = generate_pdf(query, ans_text, tips_text)
    
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="📄 Download Legal Summary (PDF)",
            data=pdf_file,
            file_name=f"Legal_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ======================================================
# KNOWLEDGE REPOSITORY
# ======================================================
with st.expander("📚 Access Compliance Knowledge Base"):
    st.info("Currently indexing: RBI Master Circular 2024, Limitation Act 1963, NBFC Fair Practice Code.")
    st.table(pd.DataFrame({
        "Legal Code": ["SARFAESI", "Sec 138 NI", "Arbitration"],
        "Focus": ["Physical Possession", "Cheque Bounce", "Civil Dispute"],
        "SLA": ["60 Days", "30 Days", "90 Days"]
    }))
