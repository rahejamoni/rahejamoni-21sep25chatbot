import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError

# =========================
# STREAMLIT SECRETS (required)
# =========================
# Set these in Streamlit Cloud -> Settings -> Secrets
# OPENAI_API_KEY, EXCEL_QA_PATH, EXCEL_LAN_PATH
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
EXCEL_QA_PATH = st.secrets.get("EXCEL_QA_PATH")
EXCEL_LAN_PATH = st.secrets.get("EXCEL_LAN_PATH")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()
if not EXCEL_QA_PATH or not EXCEL_LAN_PATH:
    st.error("EXCEL_QA_PATH or EXCEL_LAN_PATH missing in Streamlit Secrets.")
    st.stop()

# =========================
# OPENAI CLIENT
# =========================
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

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
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def safe_chat_completion(messages, model=CHAT_MODEL, temperature=0.2, max_tokens=220):
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def safe_embedding(texts, model=EMBED_MODEL):
    try:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except OpenAIError as e:
        st.error(f"OpenAI Embedding API error: {e}")
        return None
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# =========================
# LOAD EXCEL FILES (cached)
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
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in QA file: {missing}")
    return df[["id", "Question", "Answer", "Business"]]

@st.cache_data
def load_lan(path):
    df = pd.read_excel(path, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    required = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in LAN file: {missing}")
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# =========================
# EMBEDDINGS: build or load cache
# =========================
def build_or_load_embeddings(path, cache_path=EMBED_CACHE, force_refresh=False):
    qa_df = load_qa(path)
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("csv_len") == len(qa_df):
                return saved["df"], saved["embeddings"]
        except Exception:
            pass

    corpus = [f"{q} || {a}" for q, a in zip(qa_df["Question"], qa_df["Answer"])]
    vecs = safe_embedding(corpus)
    if vecs is None:
        raise RuntimeError("Embeddings generation failed.")
    vecs = np.array(vecs, dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump({"df": qa_df, "embeddings": vecs, "csv_len": len(qa_df)}, f)
    return qa_df, vecs

# =========================
# RAG retrieve
# =========================
def retrieve(query, qa_df, qa_embs, top_k=TOP_K):
    q_emb = safe_embedding([query])
    if not q_emb:
        return []
    q_vec = np.array(q_emb[0], dtype=np.float32)
    sims = np.array([cosine(q_vec, e) for e in qa_embs])
    idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in idx:
        row = qa_df.iloc[i]
        results.append({
            "id": int(row["id"]),
            "question": row["Question"],
            "answer": row["Answer"],
            "business": row["Business"],
            "score": float(sims[i])
        })
    return results

# =========================
# AGENT SUGGESTIONS (LAN)
# =========================
def suggestions_for_lan(lan_row):
    lan_id = lan_row.get("Lan Id", "")
    business = lan_row.get("Business", "")
    status = lan_row.get("Status", "")
    date = lan_row.get("Notice Sent Date", None)
    date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"

    prompt = f"""
You are a senior NBFC collections & legal strategist advising a CALLING AGENT.
Context:
- LAN ID: {lan_id}
- Business vertical: {business}
- Current status: {status}
- Last notice date: {date_str}

Task:
Provide 3 concise, practical, polite, and tactical suggestions (total ≤ 40 words)
that a collection agent should use on a call to encourage payment.
Focus on steps the agent should take (verification, tone, mention of notice, ask for commitment).
Output only bullet lines starting with "- ".
"""
    out = safe_chat_completion([{"role":"user","content":prompt}])
    if out is None:
        # fallback suggestions if API fails
        return "- Confirm if payment was recently made.\n- Remind borrower about notice and ask for payment kindly.\n- Ask for a realistic payment commitment date."
    return out

# =========================
# AGENT SUGGESTIONS (from Excel answer)
# =========================
def suggestions_for_answer(excel_answer):
    prompt = f"""
You are a senior NBFC collections strategist advising a CALLING AGENT.
Do NOT repeat the Excel answer. Use it only as context.

Context (Excel answer):
\"\"\"{excel_answer}\"\"\"

Task:
Provide 3 concise, practical, polite, tactical suggestions (total ≤ 40 words)
a collection agent can use on a call to secure payment.
Output only bullet lines starting with "- ".
"""
    out = safe_chat_completion([{"role":"user","content":prompt}])
    if out is None:
        return "- Check if payment was made.\n- Remind about notice & request payment politely.\n- Agree on a commitment date."
    return out

# =========================
# MAIN ANSWER FUNCTION
# =========================
def answer_query(query, lan_df, qa_df, qa_embs):
    # 1. If query contains LAN id -> return sentence + suggestions
    lan_match = re.search(r"\b\d{3,}\b", query)
    if lan_match:
        lan_id = lan_match.group(0)
        subset = lan_df[lan_df["Lan Id"].str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            business = row["Business"]
            status = row["Status"]
            date = row["Notice Sent Date"]
            date_str = date.strftime("%d/%m/%Y") if pd.notna(date) else "N/A"
            # Build one-line sentence as requested
            sentence = f"LAN number {lan_id} belongs to the {business} vertical and on this LAN \"{status}\" was sent on {date_str}."
            # Get suggestions tailored to LAN record
            suggestions = suggestions_for_lan(row)
            return f"{sentence}\n\nAgent Suggestions:\n{suggestions}"

    # 2. Otherwise: RAG from Excel QA -> exact Excel answer + suggestions
    contexts = retrieve(query, qa_df, qa_embs)
    if not contexts:
        return "No relevant information found in the legal staircase file."

    best = contexts[0]
    excel_answer = best["answer"]
    suggestions = suggestions_for_answer(excel_answer)
    return f"Answer (from Excel):\n{excel_answer}\n\nAgent Suggestions:\n{suggestions}"

# =========================
# LOAD DATA (once)
# =========================
try:
    qa_df, qa_embs = build_or_load_embeddings(EXCEL_QA_PATH, EMBED_CACHE)
except Exception as e:
    st.error(f"Failed to load or build embeddings: {e}")
    st.stop()

try:
    lan_df = load_lan(EXCEL_LAN_PATH)
except Exception as e:
    st.error(f"Failed to load LAN file: {e}")
    st.stop()

# =========================
# STREAMLIT UI
# =========================
st.title("📘 NBFC Legal & Collections Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Ask your question (or enter LAN id):")

if st.button("Send"):
    if query:
        result = answer_query(query, lan_df, qa_df, qa_embs)
        st.session_state.messages.append({"role":"user","content":query})
        st.session_state.messages.append({"role":"assistant","content":result})

st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
