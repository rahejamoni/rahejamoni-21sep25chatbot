import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal & Collections Intelligence Assistant",
    page_icon="📘",
    layout="wide"
)

# ======================================================
# GLOBAL CSS (COMPACT, CLEAN UI)
# ======================================================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 1.5rem; }

.card {
    background-color: #161b22;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 14px;
}

.small-text {
    color: #9ba3af;
    font-size: 14px;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 6px;
}

hr {
    border: 0.5px solid #2a2f3a;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
    <h2>📘 NBFC Legal & Collections Intelligence Assistant</h2>
    <p class="small-text">
    AI-powered decision-support system for NBFC collection agents to understand
    legal processes, loan status, and compliant recovery actions
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CONFIG
# ======================================================
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"
EMBED_CACHE = "qa_embeddings_v2.pkl"

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================================
# UTILITIES
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=180
    )
    return response.choices[0].message["content"].strip()

def embed(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in response["data"]]

def is_general_question(query: str) -> bool:
    keywords = [
        "capital", "prime minister", "president", "population",
        "india", "delhi", "define", "what is", "who is"
    ]
    q = query.lower()
    return any(k in q for k in keywords)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "question": "Question",
        "questions": "Question",
        "answer": "Answer",
        "answers": "Answer",
        "business": "Business",
        "vertical": "Business"
    })
    df["id"] = range(len(df))
    return df[["id", "Question", "Answer", "Business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype=str)
    df["Notice Sent Date"] = pd.to_datetime(
        df["Notice Sent Date"], errors="coerce", dayfirst=True
    )
    return df

def build_embeddings():
    qa_df = load_qa()
    if os.path.exists(EMBED_CACHE):
        try:
            with open(EMBED_CACHE, "rb") as f:
                saved = pickle.load(f)
            if saved.get("len") == len(qa_df):
                return qa_df, saved["emb"]
        except Exception:
            pass

    corpus = [q + " || " + a for q, a in zip(qa_df["Question"], qa_df["Answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"emb": emb, "len": len(qa_df)}, f)

    return qa_df, emb

qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# MAIN ANSWER LOGIC
# ======================================================
def answer_query(query):
    # LAN routing
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            date = r["Notice Sent Date"]
            date_str = date.strftime("%d-%m-%Y") if pd.notna(date) else "N/A"

            answer = (
                f"LAN {lan_id} belongs to **{r['Business']}** vertical. "
                f"Current status is **{r['Status']}**, notice sent on **{date_str}**."
            )
            tips = chat(f"Give 3 polite NBFC collection call suggestions:\n{answer}")
            return answer, tips

    # General knowledge routing
    if is_general_question(query):
        return chat(query), ""

    # Legal RAG
    q_vec = embed([query])[0]
    sims = [cosine(q_vec, e) for e in qa_emb]
    best_idx = int(np.argmax(sims))

    if max(sims) < 0.35:
        return chat(query), ""

    best_answer = qa_df.iloc[best_idx]["Answer"]
    tips = chat(f"Give 3 compliant call suggestions using this context:\n{best_answer}")
    return best_answer, tips

# ======================================================
# COMPACT INFO ROW (NO DUPLICATION)
# ======================================================
info_left, info_right = st.columns([2.2, 1.3])

with info_left:
    st.markdown("""
    <div class="card">
    <div class="section-title">🔍 What does this assistant do?</div>
    <ul class="small-text">
        <li>Explains NBFC legal notices & compliance steps</li>
        <li>Identifies recovery status using LAN</li>
        <li>Guides compliant customer communication</li>
    </ul>
    <p class="small-text">⚠️ For operational guidance only. Not legal advice.</p>
    </div>
    """, unsafe_allow_html=True)

with info_right:
    st.markdown("""
    <div class="card">
    <div class="section-title">⚡ Quick guide</div>
    <ul class="small-text">
        <li>Ask legal / collections questions</li>
        <li>Enter LAN ID (e.g. 22222)</li>
        <li>Ask general questions if needed</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ASK A QUESTION (VISIBLE WITHOUT SCROLL)
# ======================================================
st.markdown("""
<div class="card">
<div class="section-title">💬 Ask a Question</div>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="Type a question or enter LAN ID (e.g. 22222)",
    label_visibility="collapsed"
)

if st.button("🚀 Submit"):
    if query.strip():
        answer, tips = answer_query(query)

        st.markdown("<div class='card'><b>🧠 System Response</b></div>", unsafe_allow_html=True)
        st.success(answer)

        if tips:
            st.markdown("<div class='card'><b>🎧 Agent Compliance Suggestions</b></div>", unsafe_allow_html=True)
            st.warning(tips)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#9ba3af; font-size:13px;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence Project
</p>
""", unsafe_allow_html=True)
