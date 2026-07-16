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

def get_shuffled_questions():
    if "sample_questions" not in st.session_state:
        st.session_state.sample_questions = SAMPLE_QUESTIONS.copy()
        random.shuffle(st.session_state.sample_questions)
    return st.session_state.sample_questions

def reshuffle_questions():
    st.session_state.sample_questions = SAMPLE_QUESTIONS.copy()
    random.shuffle(st.session_state.sample_questions)

st.set_page_config(page_title="RAG Learning Lab", page_icon="🧠", layout="wide")

st.title("🧠 RAG Learning Lab")
st.caption("A hands-on demo for retrieval, ranking, and faithfulness checking.")

if "messages" not in st.session_state:
    st.session_state.messages = []

questions = get_shuffled_questions()

with st.sidebar:
    st.header("Demo controls")
    if st.button("Shuffle sample questions"):
        reshuffle_questions()
        st.rerun()

    st.subheader("Sample questions")
    for q in questions[:6]:
        if st.button(q, key=q):
            st.session_state.pending_question = q

    st.divider()
    st.subheader("System status")
    st.write(f"Backend: {API_URL}")

    st.divider()
    st.subheader("What this demo shows")
    st.markdown(
        "- Query rewriting\n"
        "- Hybrid retrieval\n"
        "- Re-ranking\n"
        "- Faithfulness verification\n"
        "- Abstention when evidence is weak"
    )

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the solar system")

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
                    timeout=60,
                )
                res.raise_for_status()
                data = res.json()

                st.write(data["answer"])

                with st.expander("Debug details"):
                    st.write("Rewritten query:", data.get("rewritten_query", "N/A"))
                    st.write("Faithful:", data.get("faithful", "N/A"))
                    st.write("Retrieval mode:", data.get("retrieval_mode", "N/A"))
                    st.write("Latency ms:", data.get("latency_ms", "N/A"))

                with st.expander("Retrieved sources"):
                    for s in data.get("sources", []):
                        st.markdown(f"**Chunk {s['chunk_id']}** — score: {s['score']:.3f}")
                        st.write(s["text"])

                st.session_state.messages.append({"role": "assistant", "content": data["answer"]})

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI at {API_URL}. Start the backend first.")
            except Exception as e:
                st.error(f"Request failed: {e}")

with right:
    st.subheader("How to use")
    st.info(
        "Use the shuffled example questions or type your own. "
        "Questions with weak evidence should abstain instead of hallucinating."
    )

    st.subheader("Expected behavior")
    st.markdown(
        "- Supported fact → grounded answer\n"
        "- Weak evidence → abstain\n"
        "- Mixed query → retrieve again or rewrite"
    )