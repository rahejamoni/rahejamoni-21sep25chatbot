import os
import re
import pickle
import io
import textwrap # Added for better PDF text wrapping
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# ... [KEEP YOUR EXISTING PAGE CONFIG AND CSS] ...

# ======================================================
# IMPROVED PDF GENERATOR (HANDLES LONG ANSWERS)
# ======================================================
def generate_pdf(query, answer, tips):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    wrap_width = 95 # Character limit per line

    # 1. Header & Branding
    p.setFillColor(colors.HexColor("#0f172a"))
    p.rect(0, height - 80, width, 80, fill=1)
    p.setStrokeColor(colors.white)
    
    p.setFont("Helvetica-Bold", 18)
    p.setFillColor(colors.white)
    p.drawString(margin, height - 45, "NBFC LEGAL INTELLIGENCE REPORT")
    
    p.setFont("Helvetica", 10)
    p.drawString(margin, height - 65, f"Issued on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Main Content Setup
    curr_y = height - 120
    p.setFillColor(colors.black)
    
    # Section: User Query
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, curr_y, "USER QUERY:")
    curr_y -= 20
    p.setFont("Helvetica", 11)
    p.drawString(margin, curr_y, f"\"{query}\"")
    curr_y -= 40

    # Section: System Interpretation (Wrapped Text)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, curr_y, "LEGAL INTERPRETATION:")
    curr_y -= 20
    
    text_obj = p.beginText(margin, curr_y)
    text_obj.setFont("Helvetica", 11)
    text_obj.setLeading(14) # Line spacing
    
    # Wrapping logic for long AI answers
    wrapped_answer = textwrap.wrap(answer, width=wrap_width)
    for line in wrapped_answer:
        text_obj.textLine(line)
    p.drawText(text_obj)
    
    # Adjust Y position based on how many lines were drawn
    curr_y = text_obj.getY() - 40

    # Section: Agent Guidance
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, curr_y, "STRATEGIC AGENT GUIDANCE:")
    curr_y -= 20
    
    tips_obj = p.beginText(margin, curr_y)
    tips_obj.setFont("Helvetica-Oblique", 11)
    tips_obj.setLeading(14)
    
    wrapped_tips = textwrap.wrap(tips_obj, width=wrap_width) # Wrap tips as well
    for line in wrapped_tips:
        tips_obj.textLine(line)
    p.drawText(tips_obj)

    # 3. Footer
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColor(colors.grey)
    p.drawCentredString(width/2, 30, "Confidential - For Internal Use Only - Created by Mohit Raheja")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ... [KEEP YOUR SIDEBAR AND BENTO CARDS] ...

# ======================================================
# QUERY LOGIC & OUTPUT
# ======================================================
query = st.chat_input("Ask a legal question or enter LAN ID...")

if query:
    with st.spinner("Analyzing through Legal Intelligence Engine..."):
        # Placeholder for your real logic (RAG / LAN lookup)
        answer_text = "The SARFAESI Act, 2002 allows banks and other financial institutions to auction residential or commercial properties (of defaulters) to recover loans. Under this Act, the lender is not required to approach a court of law to take possession of the secured asset, provided the loan is classified as a Non-Performing Asset (NPA)."
        
        agent_tips = "Check the NPA classification date. Ensure that the 60-day demand notice has been properly acknowledged by the borrower before initiating physical possession steps."

    col_ans, col_tip = st.columns([2, 1])

    with col_ans:
        st.markdown("### 🧠 System Response")
        st.markdown(f'<div class="response-card">{answer_text}</div>', unsafe_allow_html=True)

    with col_tip:
        st.markdown("### 🎧 Agent Suggestions")
        st.markdown(f'<div class="agent-card">{agent_tips}</div>', unsafe_allow_html=True)

    # DYNAMIC PDF EXPORT
    st.markdown("---")
    # We pass the real answer_text and agent_tips variables to the PDF generator
    pdf_data = generate_pdf(query, answer_text, agent_tips)
    
    st.download_button(
        label="📄 Download Legal Summary (PDF)",
        data=pdf_data,
        file_name=f"Legal_Summary_{datetime.now().strftime('%d%m%Y')}.pdf",
        mime="application/pdf",
        use_container_width=True # Advanced UI: Makes the button full width
    )
