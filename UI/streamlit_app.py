import os
import random
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

SAMPLE_QUESTIONS = [
    "What is the Great Red Spot?",
    "Which planet rotates on its side and why is that unusual?",
    "Compare Venus and Mercury in terms of temperature and atmosphere.",
    "Why is Neptune scientifically interesting beyond being the farthest planet?",
    "What is the asteroid belt and why did it not form a planet?",
    "How does the Sun produce energy?",
    "Which planet has the strongest winds and what does that imply?",
    "Why is Venus hotter than Mercury even though it is farther from the Sun?",
    "What makes Saturn’s ring system notable?",
    "What evidence suggests Mars may once have had liquid water?",
]

def init_questions():
    if "sample_questions" not in st.session_state:
        st.session_state.sample_questions = SAMPLE_QUESTIONS.copy()
        random.shuffle(st.session_state.sample_questions)

def reshuffle_questions():
    st.session_state.sample_questions = SAMPLE_QUESTIONS.copy()
    random.shuffle(st.session_state.sample_questions)

def use_question(q):
    st.session_state.pending_question = q

st.set_page_config(page_title="RAG Learning Lab", page_icon="🧠", layout="wide")
init_questions()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 RAG Learning Lab")
st.caption("A demo for query rewriting, hybrid retrieval, reranking, faithfulness verification, and PDF uploads.")

with st.sidebar:
    st.header("Controls")
    st.write(f"Backend: `{API_URL}`")

    if st.button("Shuffle sample questions"):
        reshuffle_questions()
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        if "pending_question" in st.session_state:
            del st.session_state.pending_question
        st.rerun()

    st.divider()
    st.subheader("Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_pdf is not None:
        if st.button("Ingest PDF"):
            try:
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/ingest/pdf", files=files, timeout=120)
                res.raise_for_status()
                data = res.json()
                st.success(f"Ingested PDF. Indexed chunks: {data.get('indexed_chunks', 0)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI at {API_URL}. Start the backend first.")
            except Exception as e:
                st.error(f"PDF ingest failed: {e}")

    st.divider()
    st.subheader("Sample questions")
    for i, q in enumerate(st.session_state.sample_questions[:8]):
        if st.button(q, key=f"sample_{i}_{q}"):
            use_question(q)

    st.divider()
    st.subheader("What this demo shows")
    st.markdown(
        "- Query rewriting\n"
        "- Hybrid retrieval\n"
        "- Re-ranking\n"
        "- Faithfulness verification\n"
        "- Abstention when evidence is weak\n"
        "- PDF ingestion"
    )

top_left, top_mid, top_right = st.columns([1, 1, 1], gap="medium")

with top_left:
    st.metric("Pipeline", "Hybrid RAG")

with top_mid:
    st.metric("Answer style", "Grounded")

with top_right:
    st.metric("Extra mode", "PDF Upload")

main_col, info_col = st.columns([2.2, 1], gap="large")

with main_col:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the solar system or uploaded PDFs")

    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                res = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=120,
                )
                res.raise_for_status()
                data = res.json()

                st.write(data.get("answer", ""))

                faithful = data.get("faithful", False)
                st.write("Faithfulness:", "✅" if faithful else "❌")

                with st.expander("Debug details", expanded=False):
                    st.write("Rewritten query:", data.get("rewritten_query", "N/A"))
                    st.write("Retrieval mode:", data.get("retrieval_mode", "N/A"))
                    st.write("Latency ms:", data.get("latency_ms", "N/A"))
                    st.write("Faithful:", data.get("faithful", "N/A"))
                    st.write("Context:")
                    st.code(data.get("context", ""), language="text")

                with st.expander("Retrieved sources", expanded=False):
                    for s in data.get("sources", []):
                        st.markdown(f"**Chunk {s['chunk_id']}** — score: {s['score']:.3f}")
                        st.write(s["text"])
                        if s.get("metadata"):
                            st.caption(str(s["metadata"]))

                st.session_state.messages.append(
                    {"role": "assistant", "content": data.get("answer", "")}
                )

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI at {API_URL}. Start the backend first.")
            except Exception as e:
                st.error(f"Request failed: {e}")

with info_col:
    st.subheader("How to use")
    st.info(
        "Upload a PDF in the sidebar, ingest it, then ask questions about it. "
        "Supported questions should answer from context; weak evidence should abstain."
    )

    with st.expander("Good demo flow", expanded=True):
        st.markdown(
            "1. Ask a supported question from the built-in corpus.\n"
            "2. Upload a PDF and ingest it.\n"
            "3. Ask a question from the PDF.\n"
            "4. Try an unsupported question and observe abstention."
        )

    with st.expander("Recommended demo questions", expanded=True):
        st.markdown(
            "- What is the Great Red Spot?\n"
            "- Which planet rotates on its side and why is that unusual?\n"
            "- Why is Venus hotter than Mercury even though it is farther from the Sun?\n"
            "- What evidence suggests Mars may once have had liquid water?\n"
            "- What is the asteroid belt and why did it not form a planet?"
        )

    with st.expander("System goals", expanded=False):
        st.markdown(
            "- Query rewriting.\n"
            "- Hybrid retrieval.\n"
            "- Re-ranking.\n"
            "- Faithfulness verification.\n"
            "- Retrieve again when evidence is weak."
        )