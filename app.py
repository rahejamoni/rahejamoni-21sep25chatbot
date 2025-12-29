import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError

# ======================================================
# STREAMLIT PAGE CONFIG (UI FIRST IMPRESSION)
# ======================================================
st.set_page_config(
    page_title="NBFC Legal & Collections Intelligence Assistant",
    page_icon="📘",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style='text-align:center;'>📘 NBFC Legal & Collections Intelligence Assistant</h1>
<p style='text-align:center;color:gray;font-size:16px;'>
AI-powered decision-support system for NBFC collection agents to understand
legal processes, loan status, and compliant recovery actions
</p>
<hr>
""", unsafe_allow_html=True)

# ======================================================
# HOW TO USE
# ======================================================
with st.expander("ℹ️ What can this assistant help you with?"):
    st.markdown("""
    **This assistant enables NBFC collection agents to:**
    - Understand legal notices (Pre-sale, Auction, Possession, etc.)
    - Check loan account (LAN) recovery status
    - Learn compliance timelines and next steps
    - Use polite, compliant customer conversation strategies

    ⚠️ *This is a decision-support tool and does not replace legal advice.*
    """)

# ======================================================
# CONFIG
# ======================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"

EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

# ======================================================
# OPENAI CLIENT
# ======================================================
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# UTILITIES
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def safe_chat(messages, temperature=0.2, max_tokens=200):
    try:
        res = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return None

def safe_embed(texts):
    try:
        res = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in res.data]
    except Exception:
        return None

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"questions": "Question", "answers": "Answer"})
    df["id"] = range(len(df))
    return df[["id", "Question", "Answer", "Business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(
        df["Notice Sent Date"], errors="coerce", dayfirst=True
    )
    return df

# ======================================================
# EMBEDDINGS
# ======================================================
def build_embeddings():
    qa_df = load_qa()
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "rb") as f:
            saved = pickle.load(f)
        if saved["len"] == len(qa_df):
            return qa_df, saved["emb"]

    corpus = [f"{q} || {a}" for q, a in zip(qa_df["Question"], qa_df["Answer"])]
    emb = safe_embed(corpus)
    emb = np.array(emb, dtype=np.float32)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"emb": emb, "len": len(qa_df)}, f)

    return qa_df, emb

# ======================================================
# RETRIEVAL
# ======================================================
def retrieve_answer(query, qa_df, qa_emb):
    q_emb = safe_embed([query])
    if not q_emb:
        return None

    q_vec = np.array(q_emb[0], dtype=np.float32)
    sims = [cosine(q_vec, e) for e in qa_emb]
    best_idx = int(np.argmax(sims))
    return qa_df.iloc[best_idx]["Answer"]

# ======================================================
# AGENT SUGGESTIONS
# ======================================================
def agent_suggestions(context):
    prompt = f"""
You are a senior NBFC collections strategist advising a calling agent.

Context:
{context}

Provide 3 short, polite, compliant call suggestions (≤40 words total).
Output only bullet points starting with "- ".
"""
    out = safe_chat([{"role": "user", "content": prompt}])
    if not out:
        return "- Confirm payment status\n- Remind about notice politely\n- Ask for commitment date"
    return out

# ======================================================
# MAIN LOGIC
# ======================================================
def answer_query(query):
    # LAN detection
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            date = r["Notice Sent Date"]
            date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"

            sentence = (
                f"LAN {lan_id} belongs to **{r['Business']}** vertical. "
                f"Current status **{r['Status']}**, notice sent on **{date_str}**."
            )

            suggestions = agent_suggestions(sentence)
            return sentence, suggestions

    # Legal Q&A
    excel_answer = retrieve_answer(query, qa_df, qa_emb)
    if not excel_answer:
        return "No relevant information found.", ""

    suggestions = agent_suggestions(excel_answer)
    return excel_answer, suggestions

# ======================================================
# LOAD DATA ONCE
# ======================================================
qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# USER INPUT
# ======================================================
query = st.text_input(
    "🔍 Ask a legal/process question or enter a LAN ID",
    placeholder="e.g. What is a pre-sale notice? | 123456789"
)

if st.button("🚀 Submit Query"):
    if query.strip():
        answer, tips = answer_query(query)

        st.markdown("### 🧠 System Response")
        st.success(answer)

        if tips:
            st.markdown("### 🎧 Agent Compliance Suggestions")
            st.warning(tips)
