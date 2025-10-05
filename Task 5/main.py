import streamlit as st
import os
from dotenv import load_dotenv
from app import run_agent, build_or_refresh_rag_db, CHROMA_DB_DIR, GROQ_API_KEY
  #

load_dotenv()
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="Multi-Tool AI Agent", page_icon="🤖", layout="wide")

# ------------------- HEADER ------------------- #
st.title("🤖 Multi-Tool Agent")
st.markdown(
    """
    **Tools Integrated:**  
    🧠 RAG (PDF Knowledge) ➕ Calculator 🌐 Wikipedia  
    Type your question below and let the agent decide which tool(s) to use.
    """
)

# ------------------- SIDEBAR ------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    build_db = st.button("🔄 Rebuild RAG Database")
    if build_db:
        with st.spinner("Building RAG database from PDFs..."):
            build_or_refresh_rag_db(GROQ_API_KEY)
        st.success("✅ RAG database rebuilt successfully!")

    st.markdown("---")
    st.info("📂 Make sure your PDFs are inside the `multi_tool_agent` folder.")

# ------------------- MAIN INTERFACE ------------------- #
query = st.text_area("💬 Ask a question:", placeholder="e.g., What is the warranty policy of GlobalMart TVs?")

if st.button("🚀 Run Agent"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(query)
                st.markdown("### 🧩 **Agent Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------- FOOTER ------------------- #
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Groq + Chroma + Wikipedia APIs.")
