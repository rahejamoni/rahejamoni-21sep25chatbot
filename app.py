import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# STREAMLIT SECRETS (must be set in Streamlit Cloud)
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EXCEL_QA_PATH = st.secrets["EXCEL_QA_PATH"]
EXCEL_LAN_PATH = st.secrets["EXCEL_LAN_PATH"]

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# CONFIG
# =========================
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

# =========================
# UTILITIES
# =========================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

# =========================
# LOAD QA (LEGAL STAIRCASE)
# =========================
@st.cache_data
def load_qa(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    mapping = {"questions": "Question", "anwers": "Answer", "answers": "Answer"}
    df = df.rename(columns=mapping)
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    required = {"id", "Question", "Answer", "Business"}
    if not required.issubset(df.columns):
        st.error(f"Missing Excel columns: {required - set(df.columns)}")
        st.stop()
    return df[["id", "Question", "Answer", "Business"]]

# =========================
# LOAD LAN DATA
# =========================
@st.cache_data
def load_lan(path):
    df = pd.read_excel(path, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    required = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    if not required.issubset(df.columns):
        st.error(f"Missing LAN columns: {required - set(df.columns)}")
        st.stop()
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# =========================
# EMBEDDINGS
# =========================
def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_or_load_embeddings(path, cache_path):
    df = load_qa(path)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        if saved.get("csv_len") == len(df):
            return saved["df"], saved["embeddings"]
    corpus = [f"{q} || {a}" for q, a in zip(df["Question"], df["Answer"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)
    return df, vecs

# =========================
# RAG RETRIEVAL
# =========================
def retrieve(query, df, embeddings):
    q_vec = np.array(embed_texts([query])[0])
    sims = np.array([cosine(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:TOP_K]
    return [
        {
            "id": int(df.iloc[i]["id"]),
            "question": df.iloc[i]["Question"],
            "answer": df.iloc[i]["Answer"],
            "business": df.iloc[i]["Business"],
            "score": float(sims[i])
        }
        for i in top_idx
    ]

# =========================
# AGENT SUGGESTIONS FOR EXCEL ANSWER
# =========================
def agent_suggestions_from_answer(excel_answer):
    """
    Given the exact Excel answer text, produce short (<= ~40 words)
    practical suggestions for a collection agent.
    """
    prompt = f"""
You are a senior NBFC legal + collections strategist advising a CALLING AGENT.
Do NOT repeat or modify the Excel answer. Do NOT add legal definitions.
Provide 2–4 short, practical, polite, and tactical suggestions (total ≤ 40 words)
that a collection agent can use on a call to encourage payment.
Make them specific (e.g., check payment status, mention notice, ask for commitment date).
Main Answer (for context only):
\"\"\"{excel_answer}\"\"\"
Output ONLY the suggestions as bullet points (one per line), e.g.:
- Suggestion 1
- Suggestion 2
- Suggestion 3
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# =========================
# AGENT SUGGESTIONS FOR LAN RECORD
# =========================
def agent_suggestions_for_lan(lan_row):
    """
    Create agent suggestions tailored from LAN row fields: Lan Id, Status, Business, Notice Sent Date.
    """
    lan_id = lan_row.get("Lan Id", "")
    status = lan_row.get("Status", "")
    business = lan_row.get("Business", "")
    date = lan_row.get("Notice Sent Date", None)
    date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"

    prompt = f"""
You are a senior NBFC legal + collections strategist advising a CALLING AGENT.
Context:
- LAN ID: {lan_id}
- Business: {business}
- Current Status: {status}
- Last Notice Date: {date_str}

Task:
Provide 2–4 short, practical, polite, tactical suggestions (total ≤ 40 words)
that a collection agent should follow when calling this borrower now.
Do NOT repeat the LAN summary. Output only bullet lines starting with '- '.
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# =========================
# ANSWER LOGIC
# =========================
def answer(query, lan_df, qa_df, qa_embs):
    # Check for LAN ID first
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match and lan_df is not None:
        lan_id = lan_match.group(0)
        subset = lan_df[lan_df["Lan Id"].str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0].to_dict()
            # Build LAN summary string
            date = row.get("Notice Sent Date", None)
            date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"
            lan_summary = (
                f"LAN ID: {row.get('Lan Id')}\n"
                f"Business: {row.get('Business')}\n"
                f"Status: {row.get('Status')}\n"
                f"Last Notice Date: {date_str}"
            )
            # Get suggestions tailored to this LAN record
            try:
                suggestions = agent_suggestions_for_lan(row)
            except Exception as e:
                suggestions = "- Check payment status.\n- Politely remind about notice and ask for payment."

            return f"### LAN Summary\n{lan_summary}\n\n### Agent Suggestions\n{suggestions}"

    # Otherwise use RAG over legal staircase
    try:
        contexts = retrieve(query, qa_df, qa_embs)
    except Exception as e:
        return "Error retrieving context. Check embeddings and API."

    if not contexts:
        return "No relevant information found in the legal staircase file."

    best = contexts[0]
    excel_answer = best["answer"]

    # get agent suggestions for the excel answer
    try:
        suggestions = agent_suggestions_from_answer(excel_answer)
    except Exception:
        suggestions = "- Check if payment was made.\n- Remind customer politely about legal notice."

    return f"### Answer (from Excel)\n{excel_answer}\n\n### Agent Suggestions\n{suggestions}"

# =========================
# LOAD DATA ONCE
# =========================
qa_df, qa_embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
lan_df = load_lan(EXCEL_LAN_PATH)

# =========================
# STREAMLIT UI
# =========================
st.title("📘 NBFC Legal & Collections Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Ask your question (or enter LAN id):")

if st.button("Send"):
    if query:
        result = answer(query, lan_df, qa_df, qa_embs)
        st.session_state.messages.append({"role":"user","content":query})
        st.session_state.messages.append({"role":"assistant","content":result})

st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
