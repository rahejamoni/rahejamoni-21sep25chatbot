import streamlit as st
import pandas as pd

st.title("📊 Excel File Viewer")

# Load and display lan_data.xlsx
st.subheader("LAN Data")
lan_df = pd.read_excel("lan_data.xlsx")
st.dataframe(lan_df)

# Load and display legal_staircase.xlsx
st.subheader("Legal Staircase")
legal_df = pd.read_excel("legal_staircase.xlsx")
st.dataframe(legal_df)