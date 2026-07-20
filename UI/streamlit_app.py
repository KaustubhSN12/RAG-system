import os
import random
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

SAMPLE_QUESTIONS = [
    "What is the Great Red Spot?",
    "Which planet rotates on its side and why is that unusual?",
    "Why is Venus hotter than Mercury even though it is farther from the Sun?",
    "What evidence suggests Mars may once have had liquid water?",
    "Which planet has the strongest winds and what does that imply?",
    "What is the asteroid belt and why did it not form a planet?",
    "What makes Saturn’s ring system notable?",
    "How does the Sun produce energy?",
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

st.set_page_config(
    page_title="RAG Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(90deg, #0f766e 0%, #2563eb 50%, #7c3aed 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .good {
        color: #166534;
        background: #dcfce7;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
    }
    .warn {
        color: #92400e;
        background: #fef3c7;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
    }
    .bad {
        color: #991b1b;
        background: #fee2e2;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
    }
    .section-title {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_questions()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">🧠 RAG Learning Lab</h1>
        <p style="margin:0.35rem 0 0 0; font-size:1rem;">
            Hybrid retrieval, RRF fusion, reranking, faithfulness checks, and PDF upload support.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        st.caption(f"Selected file: {uploaded_pdf.name}")
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
    st.subheader("Pipeline")
    st.markdown(
        "- Query rewriting\n"
        "- Dense retrieval\n"
        "- Sparse retrieval\n"
        "- RRF fusion\n"
        "- Re-ranking\n"
        "- Faithfulness check"
    )

main_col, info_col = st.columns([2.1, 1], gap="large")

with main_col:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about the solar system or uploaded PDFs")

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

                faithful = data.get("faithful", False)
                retrieval_mode = data.get("retrieval_mode", "unknown")
                answer = data.get("answer", "")

                st.markdown(answer)

                badge = "✅ Faithful" if faithful else "⚠️ Weak evidence"
                badge_class = "good" if faithful else "warn"
                st.markdown(f"<span class='{badge_class}'>{badge}</span>", unsafe_allow_html=True)
                st.caption(f"Retrieval mode: {retrieval_mode} | Latency: {data.get('latency_ms', 'N/A')} ms")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dense hits", len(data.get("dense_hits", [])))
                c2.metric("Sparse hits", len(data.get("sparse_hits", [])))
                c3.metric("Fused hits", len(data.get("fused_hits", [])))
                c4.metric("Reranked hits", len(data.get("reranked_hits", [])))

                tabs = st.tabs(["Dense", "Sparse", "Fused", "Reranked"])

                def render_hits(hits):
                    if not hits:
                        st.info("No hits returned.")
                        return
                    for h in hits:
                        md = h.get("metadata", {})
                        page = md.get("page_number")
                        src = md.get("source_name")
                        st.markdown(
                            f"**Chunk {h['chunk_id']}**  \n"
                            f"Score: `{h['score']:.4f}`  \n"
                            f"Source: `{src}` | Page: `{page}`"
                        )
                        st.write(h["text"])
                        st.divider()

                with tabs[0]:
                    render_hits(data.get("dense_hits", []))
                with tabs[1]:
                    render_hits(data.get("sparse_hits", []))
                with tabs[2]:
                    render_hits(data.get("fused_hits", []))
                with tabs[3]:
                    render_hits(data.get("reranked_hits", []))

                with st.expander("Final response details", expanded=False):
                    st.write("Rewritten query:", data.get("rewritten_query", "N/A"))
                    st.write("Faithful:", faithful)
                    st.write("Context:")
                    st.code(data.get("context", ""), language="text")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI at {API_URL}. Start the backend first.")
            except Exception as e:
                st.error(f"Request failed: {e}")

with info_col:
    st.markdown("### Project status")
    st.markdown(
        """
        <div class="metric-card">
            <p><b>Current focus:</b> hybrid retrieval transparency</p>
            <p><b>Next steps:</b> citations, evaluation set, deduplication</p>
            <p><b>UI goal:</b> show each retrieval stage clearly</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Good demo flow")
    st.markdown(
        "1. Ask a supported question.\n"
        "2. Inspect dense vs sparse retrieval.\n"
        "3. Compare RRF fusion and reranking.\n"
        "4. Upload a PDF and ask a document-specific question."
    )

    with st.expander("Sample prompts", expanded=True):
        st.markdown(
            "- What is the Great Red Spot?\n"
            "- Which planet rotates on its side?\n"
            "- Why is Venus hotter than Mercury?\n"
            "- What evidence suggests Mars had liquid water?"
        )

    with st.expander("Theme tip", expanded=False):
        st.markdown(
            "For a stronger visual identity, set colors in `.streamlit/config.toml`."
        )