import os
import io
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI   # ✅ correct import (fixes your error)

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

QA_FILE = "legal_staircase.xlsx"
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ======================================================
# STYLING
# ======================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left,#0f172a,#020617); color:#f8fafc; }
.card { background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
        border-radius:16px; padding:20px; }
.answer { border-left:4px solid #3b82f6; padding-left:16px; }
.agent { border-left:4px solid #22c55e; padding-left:16px; background:rgba(34,197,94,0.05); }
small { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD + EMBEDDINGS (RAG)
# ======================================================
@st.cache_resource
def load_knowledge():
    df = pd.read_excel(QA_FILE)
    df["combined"] = df["Question"].astype(str) + " " + df["Answer"].astype(str)

    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "rb") as f:
            saved = pickle.load(f)
        if saved["len"] == len(df):
            return df, saved["emb"]

    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=df["combined"].tolist()
    ).data
    emb = np.array([e.embedding for e in emb])

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"len": len(df), "emb": emb}, f)

    return df, emb

qa_df, qa_emb = load_knowledge()

# ======================================================
# PDF DOWNLOAD
# ======================================================
def generate_pdf(question, answer, advice):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    w, h = letter

    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, h - 40, "NBFC Legal Intelligence Report")

    p.setFont("Helvetica", 10)
    p.drawString(40, h - 60, f"Generated: {datetime.now()}")

    y = h - 100
    p.setFont("Helvetica-Bold", 11)
    p.drawString(40, y, "Question:")
    y -= 16
    p.setFont("Helvetica", 11)
    p.drawString(40, y, question)

    y -= 30
    p.setFont("Helvetica-Bold", 11)
    p.drawString(40, y, "System Answer:")
    y -= 16
    for line in answer.split("\n"):
        p.drawString(40, y, line)
        y -= 14

    y -= 20
    p.setFont("Helvetica-Bold", 11)
    p.drawString(40, y, "Agent Advice:")
    y -= 16
    for line in advice.split("\n"):
        p.drawString(40, y, line)
        y -= 14

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## 🧠 Intel Center")

    st.markdown("""
**What does this assistant do?**
- Explains NBFC legal notices & recovery stages  
- Interprets SARFAESI, Section 138 & arbitration  
- Fetches LAN-level recovery status  
- Suggests compliant customer communication  

⚠️ *Operational guidance only*
""")

    st.markdown("""
**ℹ️ How to use**
- Ask a legal / collections question  
- Enter a LAN ID (e.g. 22222)  
- Review system answer & agent advice
""")

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🛡️ NBFC Legal Intelligence Hub")
st.caption("AI-powered decision support for collections & legal compliance")

# ======================================================
# FEATURE BOXES
# ======================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='card'><h4>⚖️ Legal Staircase</h4><small>SARFAESI • Sec 138 • Arbitration</small></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><h4>🔍 LAN Intelligence</h4><small>Loan status • DPD • Notices</small></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><h4>📞 Communication</h4><small>Compliant scripts & guidance</small></div>", unsafe_allow_html=True)

# ======================================================
# CHAT HISTORY
# ======================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ======================================================
# QUERY INPUT
# ======================================================
query = st.chat_input("Ask legal question or enter LAN ID...")

if query:
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    sims = cosine_similarity([q_emb], qa_emb)[0]
    idx = np.argmax(sims)

    use_rag = sims[idx] > 0.35

    if use_rag:
        context = qa_df.iloc[idx]["Answer"]
    else:
        context = ""

    system_prompt = f"""
You are an NBFC legal intelligence assistant.

If context is provided, answer ONLY from it.
If no context, answer using general knowledge.

Always add a short 'Agent Advice' section.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    answer = response.choices[0].message.content

    agent_advice = (
        "• Ensure communication is polite and RBI-compliant\n"
        "• Clearly explain consequences of non-payment\n"
        "• Document all calls, SMS, and visits\n"
        "• Avoid coercive or threatening language"
    )

    st.session_state.chat.append((query, answer, agent_advice))

# ======================================================
# DISPLAY CHAT
# ======================================================
for q, a, adv in st.session_state.chat[::-1]:
    st.markdown(f"<div class='card answer'><h4>💡 Legal / System Answer</h4>{a}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card agent'><h4>🎧 Agent Advice</h4>{adv}</div>", unsafe_allow_html=True)

    pdf = generate_pdf(q, a, adv)
    st.download_button(
        "📄 Download Response (PDF)",
        pdf,
        file_name="nbfc_legal_response.pdf",
        mime="application/pdf"
    )
