import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBFC Legal & Collections Intelligence Assistant",
    page_icon="📘",
    layout="wide"
)

# ======================================================
# CUSTOM CSS (THIS IS THE GAME CHANGER)
# ======================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #ffffff;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.small-text {
    color: #9ba3af;
    font-size: 15px;
}
hr {
    border: 1px solid #2a2f3a;
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
# INTRODUCTION
# ======================================================
st.markdown("""
<div class="card">
<h3>🔍 What does this assistant do?</h3>
<ul class="small-text">
<li>Explains NBFC legal notices (Pre-sale, Auction, Possession, etc.)</li>
<li>Identifies recovery status using Loan Account Number (LAN)</li>
<li>Guides agents on compliance timelines</li>
<li>Suggests polite, compliant customer communication</li>
</ul>
<p class="small-text">
⚠️ This tool assists agents and does not replace legal advice.
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CONFIG
# ======================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
QA_FILE = "legal_staircase.xlsx"
LAN_FILE = "lan_data.xlsx"

EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ======================================================
# OPENAI CLIENT
# ======================================================
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# UTIL FUNCTIONS
# ======================================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def chat(messages):
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=200
    )
    return res.choices[0].message.content.strip()

def embed(texts):
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_qa():
    df = pd.read_excel(QA_FILE)
    df.columns = df.columns.str.strip()
    df["id"] = range(len(df))
    return df[["id", "Question", "Answer", "Business"]]

@st.cache_data
def load_lan():
    df = pd.read_excel(LAN_FILE, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    df["Lan Id"] = df["Lan Id"].astype(str)
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], errors="coerce")
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

    corpus = [q + " || " + a for q, a in zip(qa_df["Question"], qa_df["Answer"])]
    emb = np.array(embed(corpus), dtype=np.float32)

    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"emb": emb, "len": len(qa_df)}, f)

    return qa_df, emb

# ======================================================
# ANSWER LOGIC
# ======================================================
def answer_query(query):
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        row = lan_df[lan_df["Lan Id"] == lan_id]
        if not row.empty:
            r = row.iloc[0]
            date = r["Notice Sent Date"]
            d = date.strftime("%d-%m-%Y") if pd.notna(date) else "N/A"

            answer = (
                f"LAN {lan_id} belongs to **{r['Business']}** vertical. "
                f"Current status is **{r['Status']}**, notice sent on **{d}**."
            )

            tips = chat([{
                "role": "user",
                "content": f"Give 3 polite, compliant call suggestions for this case: {answer}"
            }])

            return answer, tips

    # Legal question
    q_vec = embed([query])[0]
    sims = [cosine(q_vec, e) for e in qa_emb]
    best = qa_df.iloc[int(np.argmax(sims))]["Answer"]

    tips = chat([{
        "role": "user",
        "content": f"Give 3 polite NBFC collection call suggestions using this context: {best}"
    }])

    return best, tips

# ======================================================
# LOAD DATA
# ======================================================
qa_df, qa_emb = build_embeddings()
lan_df = load_lan()

# ======================================================
# QUERY INPUT
# ======================================================
st.markdown("""
<div class="card">
<h3>💬 Ask a Question</h3>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="e.g. What is a pre-sale notice?  |  Enter LAN ID"
)

if st.button("🚀 Submit"):
    if query.strip():
        answer, tips = answer_query(query)

        st.markdown("""
        <div class="card">
        <h3>🧠 System Response</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success(answer)

        st.markdown("""
        <div class="card">
        <h3>🎧 Agent Compliance Suggestions</h3>
        </div>
        """, unsafe_allow_html=True)

        st.warning(tips)
