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
# GLOBAL CSS (CLEAN, ENTERPRISE LOOK)
# ======================================================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
}

.small-text {
    color: #9ba3af;
    font-size: 14.5px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
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
    <h1>📘 NBFC Legal & Collections Intelligence Assistant</h1>
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
        max_tokens=200
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
        "country", "india", "delhi", "define", "what is", "who is"
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
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], errors="coerce", dayfirst=True)
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
# INFORMATION SECTION (FULL-WIDTH UTILIZATION)
# ======================================================
left_info, right_info = st.columns([2.2, 1.3])

with left_info:
    st.markdown("""
    <div class="card">
    <div class="section-title">🔍 What does this assistant do?</div>
    <ul class="small-text">
        <li>Explains NBFC legal notices (Pre-sale, Auction, Possession, etc.)</li>
        <li>Identifies recovery status using Loan Account Number (LAN)</li>
        <li>Guides agents on compliance timelines</li>
        <li>Answers general knowledge questions when applicable</li>
        <li>Suggests polite, compliant customer communication</li>
    </ul>
    <p class="small-text">⚠️ This tool assists agents and does not replace legal advice.</p>
    </div>
    """, unsafe_allow_html=True)

with right_info:
    st.markdown("""
    <div class="card">
    <div class="section-title">ℹ️ How to use</div>
    <ul class="small-text">
        <li>Ask NBFC legal or collections questions</li>
        <li>Enter a LAN ID to check recovery status</li>
        <li>Ask general questions (capital, definitions)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-title">🧪 Example Queries</div>
    <ul class="small-text">
        <li>What is a pre-sale notice?</li>
        <li>What is the capital of India?</li>
        <li>123456789</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-title">⚙️ System Capabilities</div>
    <ul class="small-text">
        <li>Hybrid AI routing (RAG + LLM fallback)</li>
        <li>LAN-based rule intelligence</li>
        <li>Compliance-safe suggestions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# QUERY SECTION
# ======================================================
st.markdown("""
<div class="card">
<div class="section-title">💬 Ask a Question</div>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="e.g. What is a pre-sale notice? | What is the capital of India? | Enter LAN ID"
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
<p style="text-align:center; color:#9ba3af; font-size:14px;">
Created by <b>Mohit Raheja</b> | Applied AI & Decision Intelligence Project
</p>
""", unsafe_allow_html=True)
