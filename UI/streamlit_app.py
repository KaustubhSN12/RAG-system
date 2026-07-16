import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chat", page_icon="🤖")
st.title("RAG Chat")

question = st.chat_input("Ask a question about the knowledge base")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            res = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            res.raise_for_status()
            data = res.json()

            st.write(data["answer"])

            with st.expander("Retrieved sources"):
                for s in data["sources"]:
                    st.markdown(f"**Chunk {s['chunk_id']}** — score: {s['score']:.3f}")
                    st.write(s["text"])

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI at {API_URL}. Start the backend first.")
        except Exception as e:
            st.error(f"Request failed: {e}")