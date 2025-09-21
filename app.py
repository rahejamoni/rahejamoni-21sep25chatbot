import streamlit as st
from openai import OpenAI
import os

# ==============================
# Setup
# ==============================
st.title("🤖 Simple AI Chatbot")

# Load API key securely
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_api_key:
    st.error("Please add your OpenAI API key in Streamlit secrets!")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# Chat input
# ==============================
user_input = st.text_input("Ask me anything:")

if st.button("Send"):
    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response from OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",  # lightweight model
                messages=st.session_state.messages
            )
            bot_reply = response.choices[0].message.content

            # Save bot reply
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# Display chat history
# ==============================
st.subheader("Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
