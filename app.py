import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# LOAD STREAMLIT SECRETS
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
MAX_WORDS = 150

# =========================
# UTILITIES
# =========================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# =========================
# LOAD LEGAL STAIRCASE EXCEL
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
# EMBEDDING FUNCTIONS
# =========================
def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_or_load_embeddings(path, cache_path):
    df = load_qa(path)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)

        if saved["csv_len"] == len(df):
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

    results = []
    for i in top_idx:
        results.append({
            "id": int(df.iloc[i]["id"]),
            "question": df.iloc[i]["Question"],
            "answer": df.iloc[i]["Answer"],
            "business": df.iloc[i]["Business"],
            "score": float(sims[i])
        })
    return results


# =========================
# SUGGESTIONS FOR COLLECTION AGENT
# =========================
def agent_suggestions(excel_answer):
    prompt = f"""
You are a senior NBFC legal + collections strategist.

Rules:
- DO NOT repeat the Excel answer.
- Provide 2–3 short but powerful suggestions.
- Total length must be **under 40 words**.
- Suggestions should help a COLLECTION AGENT convince the borrower to pay.
- Suggestions must be practical, polite, and action-oriented.
- Examples of style:
  • "Politely check if any recent payment was made before escalating."
  • "Remind borrower that legal action is active and encourage quick resolution."
  • "Stay calm and guide customer toward a realistic commitment date."

Excel Answer (do NOT repeat):
\"\"\"{excel_answer}\"\"\"

Now give ONLY the suggestions section:

Suggestions:
- ...
- ...
- ...
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


# =========================
# ANSWER QUERY
# =========================
def answer(query, lan_df, qa_df, qa_embs):

    # --- If query contains a LAN ID ---
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        subset = lan_df[lan_df["Lan Id"] == lan_id]

        if not subset.empty:
            row = subset.iloc[0]
            date = row["Notice Sent Date"]
            date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"

            return (
                f"**LAN ID:** {row['Lan Id']}\n"
                f"**Business:** {row['Business']}\n"
                f"**Status:** {row['Status']}\n"
                f"**Last Notice Date:** {date_str}"
            )

    # --- Otherwise do RAG from Excel ---
    ctx = retrieve(query, qa_df, qa_embs)
    best = ctx[0]

    excel_answer = best["answer"]
    suggestions = agent_suggestions(excel_answer)

    final_output = (
        f"### ✅ Answer (From Excel)\n{excel_answer}\n\n"
        f"### 📌 Agent Suggestions\n{suggestions}"
    )

    return final_output


# =========================
# LOAD DATA
# =========================
qa_df, qa_embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
lan_df = load_lan(EXCEL_LAN_PATH)

# =========================
# STREAMLIT UI
# =========================
st.title("📘 NBFC Legal & Collections Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Ask your question:")

if st.button("Send"):
    if query:
        result = answer(query, lan_df, qa_df, qa_embs)

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": result})

st.subheader("Chat History")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
