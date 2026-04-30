"""
HealthRAG-IN — Streamlit UI
============================
A glassmorphism-themed chat interface for the medical RAG system.

Run from project root:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.generation.answer import RAGPipeline


# ──────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must come before any other Streamlit calls)
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HealthRAG-IN",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────
#  CUSTOM CSS  (the entire glassmorphism theme lives here)
# ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Import bold display font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background: deep navy radial gradient */
.stApp {
    background:
        radial-gradient(ellipse at top left, #0a3a6e 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, #1B7FDC 0%, transparent 60%),
        linear-gradient(180deg, #0a1929 0%, #0f2a47 50%, #1a3a5e 100%);
    background-attachment: fixed;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

/* Main content container */
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1100px !important;
}

/* HEADER ─────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}
.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0DB8D3 0%, #1B7FDC 50%, #ffffff 100%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.app-subtitle {
    font-size: 1.1rem;
    color: #8FB5D5;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* GLASS CARD ─────────────────────────────────────────────── */
.glass-card {
    background: rgba(13, 184, 211, 0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(27, 127, 220, 0.25);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* Question / answer cards with glow */
.user-message {
    background: linear-gradient(135deg, rgba(27, 127, 220, 0.18) 0%, rgba(13, 184, 211, 0.08) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(13, 184, 211, 0.3);
    border-radius: 18px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px rgba(13, 184, 211, 0.1);
    color: #E8F4FD;
    font-weight: 500;
    font-size: 1.05rem;
}
.assistant-message {
    background: rgba(15, 42, 71, 0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(27, 127, 220, 0.2);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    color: #E8F4FD;
    line-height: 1.7;
    font-size: 1rem;
}

/* Section labels */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #0DB8D3;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Citation chip */
.citation-chip {
    display: inline-block;
    background: rgba(13, 184, 211, 0.18);
    border: 1px solid rgba(13, 184, 211, 0.5);
    color: #0DB8D3;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 0 2px;
}

/* Source pill (sidebar) */
.source-pill {
    background: rgba(13, 184, 211, 0.06);
    border: 1px solid rgba(27, 127, 220, 0.2);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    transition: all 0.25s ease;
}
.source-pill:hover {
    background: rgba(13, 184, 211, 0.12);
    border-color: rgba(13, 184, 211, 0.5);
    transform: translateY(-2px);
}
.source-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: linear-gradient(135deg, #0DB8D3, #1B7FDC);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.8rem;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 12px rgba(13, 184, 211, 0.5);
}
.source-source {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    color: #0DB8D3;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.source-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: #E8F4FD;
    margin-top: 4px;
    line-height: 1.3;
}
.source-link {
    font-size: 0.78rem;
    color: #8FB5D5;
    text-decoration: none;
    margin-top: 4px;
    display: inline-block;
}
.source-link:hover { color: #0DB8D3; }

/* Provider badge */
.provider-badge {
    display: inline-block;
    background: rgba(46, 125, 50, 0.18);
    border: 1px solid rgba(46, 125, 50, 0.4);
    color: #7DD8A4;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.provider-badge.gemini {
    background: rgba(183, 121, 31, 0.18);
    border-color: rgba(183, 121, 31, 0.4);
    color: #FFD180;
}

/* Stats card */
.stat-card {
    background: rgba(13, 184, 211, 0.05);
    border: 1px solid rgba(27, 127, 220, 0.2);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0DB8D3;
    line-height: 1;
}
.stat-label {
    font-size: 0.7rem;
    color: #8FB5D5;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

/* INPUT BOX ──────────────────────────────────────────────── */
.stChatInput {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(15, 42, 71, 0.5) !important;
    border: 1px solid rgba(27, 127, 220, 0.3) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #0DB8D3 !important;
    box-shadow: 0 0 24px rgba(13, 184, 211, 0.25) !important;
}

/* SIDEBAR ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 25, 41, 0.85) !important;
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border-right: 1px solid rgba(27, 127, 220, 0.15);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}

/* WARNING / EMERGENCY ────────────────────────────────────── */
.emergency-banner {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(220, 38, 38, 0.08));
    border: 1px solid rgba(220, 38, 38, 0.5);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    color: #FFB4B4;
    font-weight: 600;
    margin-bottom: 1rem;
    box-shadow: 0 0 24px rgba(220, 38, 38, 0.15);
}

/* SCROLLBAR ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(15, 42, 71, 0.3); }
::-webkit-scrollbar-thumb {
    background: rgba(13, 184, 211, 0.4);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(13, 184, 211, 0.6); }

/* SPINNER ──────────────────────────────────────────────── */
.thinking {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 1.5rem;
    background: rgba(13, 184, 211, 0.06);
    border: 1px solid rgba(13, 184, 211, 0.2);
    border-radius: 14px;
    color: #8FB5D5;
    font-weight: 500;
}
.thinking-dot {
    width: 8px;
    height: 8px;
    background: #0DB8D3;
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
    40% { opacity: 1; transform: scale(1.05); }
}

/* Hide Streamlit's default chat avatar circles  */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  RAG PIPELINE  (cached so it loads only once)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the RAG pipeline once and reuse across queries."""
    return RAGPipeline()


# ──────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">HealthRAG-IN</div>
        <div class="app-subtitle">
            Grounded Medical Q&amp;A • WHO • PubMed • ICMR
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="section-label">About</div>
        <div class="glass-card" style="margin-bottom:1rem;">
            <div style="font-size:0.95rem; line-height:1.6; color:#E8F4FD;">
            A retrieval-augmented Q&amp;A system for diabetes-related medical questions,
            grounded in <b>WHO fact sheets</b>, <b>PubMed research</b>,
            and <b>ICMR clinical guidelines</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Corpus Stats</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="stat-card"><div class="stat-value">490</div>'
            '<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="stat-card"><div class="stat-value">1,247</div>'
            '<div class="stat-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Try Asking</div>', unsafe_allow_html=True)

    sample_questions = [
        "What is HbA1c and why does it matter?",
        "What diet do ICMR guidelines recommend for diabetics in India?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "How is diabetic ketoacidosis treated?",
    ]

    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-label">Disclaimer</div>
        <div class="glass-card" style="font-size:0.82rem; color:#8FB5D5; line-height:1.5;">
        Educational tool only. Not a substitute for professional medical advice.
        Always consult a qualified physician for personal health decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────
#  CHAT STATE
# ──────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" in st.session_state:
    pending = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": pending, "process": True})
    st.rerun()


# ──────────────────────────────────────────────────────────────────
#  RENDER PAST MESSAGES
# ──────────────────────────────────────────────────────────────────

def render_user_message(content):
    st.markdown(
        f'<div class="section-label">You asked</div>'
        f'<div class="user-message">{content}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_message(result):
    """Render the assistant's structured response with citations."""
    answer = result["answer"]
    chunks = result.get("retrieved_chunks", [])
    provider = result.get("model_name", "unknown")
    is_emergency = result.get("is_emergency", False)
    rag_time = result.get("total_time_sec", 0)

    if is_emergency:
        st.markdown(
            '<div class="emergency-banner">⚠ Possible medical emergency detected. '
            'Please call your local emergency number or go to the nearest hospital immediately.</div>',
            unsafe_allow_html=True,
        )

    # Convert [N] citations to styled chips
    import re
    def chip(m):
        return f'<span class="citation-chip">{m.group(0)}</span>'
    rendered_answer = re.sub(r"\[[\d,\s]+\]", chip, answer)
    rendered_answer = rendered_answer.replace("\n", "<br>")

    st.markdown('<div class="section-label">HealthRAG-IN says</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-message">{rendered_answer}</div>',
                unsafe_allow_html=True)

    # Provider + latency badge row
    badge_class = "provider-badge gemini" if provider == "gemini" else "provider-badge"
    badge_label = provider.upper() if provider != "none" else "FALLBACK"
    st.markdown(
        f'<div style="margin: -8px 0 1rem 0; display:flex; gap:10px;">'
        f'<span class="{badge_class}">{badge_label}</span>'
        f'<span class="provider-badge" style="background:rgba(143,181,213,0.12); '
        f'border-color:rgba(143,181,213,0.3); color:#8FB5D5;">'
        f'{rag_time:.1f}s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sources
    if chunks:
        with st.expander(f"📚 Sources ({len(chunks)})", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                source = meta.get("source", "Unknown")
                title = meta.get("parent_title", "Untitled")
                url = meta.get("parent_url", "")
                excerpt = chunk.get("text", "")[:280]

                link_html = (
                    f'<a class="source-link" href="{url}" target="_blank">{url}</a>'
                    if url else ""
                )
                st.markdown(
                    f'''<div class="source-pill">
                        <span class="source-num">{i}</span>
                        <span class="source-source">{source}</span>
                        <div class="source-title">{title}</div>
                        <div style="color:#B0C8DD; font-size:0.85rem; margin-top:8px; line-height:1.5;">
                            {excerpt}…
                        </div>
                        {link_html}
                    </div>''',
                    unsafe_allow_html=True,
                )


# Render historical messages first
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    elif msg["role"] == "assistant" and msg.get("result"):
        render_assistant_message(msg["result"])


# ──────────────────────────────────────────────────────────────────
#  PROCESS PENDING QUESTION
# ──────────────────────────────────────────────────────────────────

# Find the most recent unprocessed user message
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg["role"] == "user" and last_msg.get("process"):
    last_msg["process"] = False  # mark as being processed

    # Show thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        '<div class="thinking">'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        '<span style="margin-left:8px;">Searching medical sources and generating answer…</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        rag = get_pipeline()
        result = rag.ask(last_msg["content"])
        thinking_placeholder.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "result": result,
        })
        st.rerun()

    except Exception as e:
        thinking_placeholder.empty()
        st.markdown(
            f'<div class="emergency-banner">⚠ Sorry, something went wrong: {e}</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────
#  CHAT INPUT
# ──────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a medical question about diabetes…")
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "process": True,
    })
    st.rerun()