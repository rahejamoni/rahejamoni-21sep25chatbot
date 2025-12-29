import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="NBFC Intel | Legal & Collections",
    page_icon="🛡️",
    layout="wide",
)

# ======================================================
# ADVANCED CUSTOM CSS (Glassmorphism & Professional UI)
# ======================================================
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Bento Box Cards */
    .bento-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .bento-card:hover {
        border: 1px solid rgba(0, 160, 240, 0.4);
        transform: translateY(-2px);
    }

    /* Typography */
    .hero-text {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Status Tags */
    .status-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(45, 212, 191, 0.1);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.2);
    }

    /* Custom Chat Container */
    .response-area {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Input Styling */
    .stTextInput input {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR NAVIGATION & STATS
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=80)
    st.markdown("## Intel Center")
    st.markdown("---")
    
    # Dynamic Stats
    st.markdown("### 📊 Live Analytics")
    st.metric(label="Compliance Score", value="98.2%", delta="0.4%")
    st.metric(label="Case Resolution", value="1.4k", delta="12% vs LY")
    
    st.markdown("---")
    st.caption("v2.4.0 High-Performance Edition")
    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

# ======================================================
# MAIN HERO SECTION
# ======================================================
st.markdown('<h1 class="hero-text">Legal Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Advanced decision-support for NBFC collections and legal risk mitigation.</p>', unsafe_allow_html=True)

# Bento Grid for Features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#3b82f6;">⚖️ Legal Staircase</h4>
        <p style="font-size: 0.9rem; color: #94a3b8;">Instant interpretation of Section 138, Sarfaesi, and Arbitration protocols.</p>
        <span class="status-tag">Updated: RBI 2024</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#2dd4bf;">🔍 LAN Intelligence</h4>
        <p style="font-size: 0.9rem; color: #94a3b8;">Real-time loan status, notice history, and debtor behavioral flags.</p>
        <span class="status-tag">Active Database</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="bento-card">
        <h4 style="color:#f59e0b;">📞 Communication</h4>
        <p style="font-size: 0.9rem; color: #94a3b8;">Compliant call scripts generated dynamically based on case severity.</p>
        <span class="status-tag">Audit-Ready</span>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# SEARCH & QUERY ENGINE
# ======================================================
st.markdown("### 🧬 Query Intelligence")
query = st.chat_input("Ask a question (e.g., 'What is the limitation period for vehicle repossession?')")

if query:
    # Logic Processing Placeholder
    # (In your real app, call your answer_query function here)
    with st.spinner("Processing through Legal Engine..."):
        # Simulated logic for the demo
        is_lan = re.search(r"\b\d{3,}\b", query)
        
        # Display Area
        st.markdown("---")
        res_col, tip_col = st.columns([2, 1])
        
        with res_col:
            st.markdown("#### 💡 Intelligence Output")
            st.markdown(f"""
            <div class="response-area">
                <strong>Result for:</strong> <em>"{query}"</em><br><br>
                Based on current NBFC guidelines, the requested action is compliant under the <strong>Master Circular on Loans and Advances</strong>. 
                The Limitation Act specifies a 3-year window for recovery suits from the date of default.
            </div>
            """, unsafe_allow_html=True)
            
        with tip_col:
            st.markdown("#### 🛠️ Action Plan")
            st.warning("**Step 1:** Verify the DPD (Days Past Due) bucket.")
            st.info("**Step 2:** Ensure the 15-day Pre-sale Notice is acknowledged.")
            st.success("**Step 3:** Use Script #14 for high-intent communication.")

# ======================================================
# INTERACTIVE DATA TABS (Below the Chat)
# ======================================================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📋 Recent Case Logs", "📑 Compliance Knowledge Base"])

with tab1:
    # Example of how to show your LAN dataframe professionally
    mock_data = pd.DataFrame({
        "LAN ID": ["10293", "10294", "10295"],
        "Notice Type": ["Sec 138", "Pre-Sale", "Demand Notice"],
        "Status": ["Delivered", "In-Transit", "Expired"],
        "Risk Level": ["High", "Medium", "Low"]
    })
    st.dataframe(mock_data, use_container_width=True)

with tab2:
    st.markdown("""
    * **SARFAESI Act:** Procedure for enforcing security interest.
    * **Section 138:** Negotiable Instruments Act (Cheque Bounce).
    * **RBI Fair Practices Code:** Guidelines for collection agents.
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
    <div style="margin-top: 50px; padding: 20px; text-align: center; border-top: 1px solid rgba(255,255,255,0.05);">
        <p style="color: #64748b; font-size: 0.8rem;">
            Designed by <strong>Mohit Raheja</strong> | Applied AI Division<br>
            Secure Enterprise Instance - 128-bit Encryption Active
        </p>
    </div>
""", unsafe_allow_html=True)
