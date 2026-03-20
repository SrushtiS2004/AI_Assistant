import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.config import GROQ_API_KEY
from models.llm import generate_response
from utils.rag import create_vector_store, retrieve, clear_vector_store
from utils.web_search import search_web
from utils.helpers import process_uploaded_file

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Interview Coach", page_icon="🧠", layout="wide")

USER_AVATAR = "👤"
BOT_AVATAR  = "🤖"

# ── Dark theme constants ───────────────────────────────────────────────────────
BG_COLOR      = "#212121"
CARD_BG       = "#2F2F2F"
TEXT_COLOR    = "#ECECEC"
MUTED_TEXT    = "#9B9B9B"
BORDER_COLOR  = "#424242"
USER_BUBBLE   = "#343541"
PRIMARY_COLOR = "#10A37F"
HOVER_COLOR   = "#3A3A3A"

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main > div {{
    background-color: {BG_COLOR} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: {TEXT_COLOR} !important;
}}

/* Enough padding so the last chat message never hides under the sticky input bar */
[data-testid="stMain"] > div:first-child,
section.main > div:first-child,
.main > div {{
    padding-bottom: 300px !important;
}}

/* Make the sticky bottom bar float cleanly */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {{
    background-color: {BG_COLOR} !important;
    padding-top: 10px !important;
}}

h1,h2,h3,h4,h5,h6,p,span,div,label,li,td,th,caption {{
    font-family: 'Inter', system-ui, sans-serif !important;
    color: {TEXT_COLOR} !important;
}}

/* ── app header ── */
.app-header {{ padding: 14px 0 2px 0; }}
.app-header h1 {{
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    color: {TEXT_COLOR} !important;
    letter-spacing: -0.02em;
}}
.app-header p {{
    margin: 4px 0 8px 0;
    color: {MUTED_TEXT} !important;
    font-size: 0.93rem;
}}

/* ── status badge ── */
.source-badge {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
    letter-spacing: 0.04em;
}}
.badge-web {{
    background-color: #252b3b;
    color: #93a8d4 !important;
    border: 1px solid #3a4560;
}}
.badge-doc {{
    background-color: #1a3050;
    color: #93c5fd !important;
    border: 1px solid #2d4f7a;
}}

/* ── chat bubbles ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    background-color: transparent !important;
    padding: 10px 4px;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background-color: {USER_BUBBLE} !important;
    border-radius: 14px !important;
    padding: 12px 16px;
    margin-left: auto;
    width: fit-content;
    max-width: 78%;
    flex-direction: row-reverse;
    text-align: right;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="chatAvatarIcon-user"] {{
    margin-left: 10px;
    margin-right: 0;
}}

/* ── chat input ── */
.stChatInputContainer {{
    background-color: {CARD_BG} !important;
    border-radius: 14px !important;
    border: 1px solid {BORDER_COLOR} !important;
    padding: 2px 4px !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.4);
}}
.stChatInputContainer textarea,
.stChatInputContainer textarea::placeholder {{
    color: {MUTED_TEXT} !important;
    background-color: transparent !important;
}}
textarea[data-testid="stChatInputTextArea"] {{
    color: {TEXT_COLOR} !important;
}}

/* ── sidebar cards ── */
.s-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 16px;
}}
.s-card-title {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {MUTED_TEXT} !important;
    margin-bottom: 12px;
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 7px;
}}

/* ── buttons ── */
.stButton > button {{
    border-radius: 9px !important;
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
    color: {TEXT_COLOR} !important;
    font-weight: 500 !important;
    transition: 0.15s;
}}
.stButton > button:hover {{
    background-color: {HOVER_COLOR} !important;
    border-color: {PRIMARY_COLOR} !important;
}}
button[kind="primary"] {{
    background-color: {PRIMARY_COLOR} !important;
    border: none !important;
    color: #fff !important;
}}
button[kind="primary"] p {{ color: #fff !important; }}
button[kind="primary"]:hover {{ opacity: 0.88 !important; }}

/* ── radio ── */
[data-testid="stRadio"] label {{ color: {TEXT_COLOR} !important; }}

/* ── file uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploadDropzone"] {{
    background-color: {BG_COLOR} !important;
    border: 1px dashed {BORDER_COLOR} !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploader"] * {{ color: {TEXT_COLOR} !important; }}
[data-testid="stFileUploader"] button {{
    background-color: {PRIMARY_COLOR} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
}}
[data-testid="stFileUploader"] button:hover {{ opacity: 0.8 !important; }}

/* ── expanders ── */
[data-testid="stExpander"] {{
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 10px !important;
}}
.streamlit-expanderHeader {{ color: {TEXT_COLOR} !important; }}

/* ── scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_COLOR}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 4px; }}

/* ── hide Streamlit chrome ── */
header, footer, #MainMenu, [data-testid="stSidebar"] {{
    visibility: hidden; display: none; height: 0;
}}
</style>
"""

# ── Session defaults ──────────────────────────────────────────────────────────
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "👋 **Welcome! I'm your AI Interview Coach.**\n\n"
                "Here is how I work:\n"
                "- 📄 **Upload your resume or JD** (right panel) → I answer specifically based on it.\n"
                "- 🌐 **No document?** I automatically search the web for up-to-date information.\n\n"
                "Ask me anything — interview Q&A, mock sessions, salary tips, resume reviews!"
            )
        }]
    if "detail_mode" not in st.session_state:
        st.session_state.detail_mode = "Detailed"

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    # Coach Settings
    st.markdown("<div class='s-card'><div class='s-card-title'>⚙️ Coach Settings</div>", unsafe_allow_html=True)
    st.radio("style", ["Concise", "Detailed"], horizontal=True, key="detail_mode", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # Knowledge Base
    st.markdown("<div class='s-card'><div class='s-card-title'>📚 Knowledge Base</div>", unsafe_allow_html=True)
    st.caption("PDF or TXT — resume, JD, or study notes")
    uploaded_file = st.file_uploader("doc", type=["txt", "pdf"], label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Process", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Indexing…"):
                    text = process_uploaded_file(uploaded_file)
                    if text and not text.startswith("Error"):
                        st.success("Indexed! ✅") if create_vector_store(text) else st.error("Failed ❌")
                    else:
                        st.error(f"Error: {text}")
            else:
                st.warning("No file selected ⚠️")
    with c2:
        if st.button("Clear", use_container_width=True):
            clear_vector_store()
            st.success("Cleared 🧹")

    st.markdown("<hr style='border-color:{};margin:12px 0'>".format(BORDER_COLOR), unsafe_allow_html=True)
    st.markdown("<div class='s-card-title'>🔍 Search Mode</div>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:{};font-size:0.85rem;'>"
        "When no document is uploaded, I automatically run a live <b>DuckDuckGo web search</b> "
        "and use those results to answer your question."
        "</span>".format(MUTED_TEXT),
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────────
def render_chat_and_input():
    st.markdown("""
        <div class='app-header'>
            <h1>🧠 AI Interview Coach</h1>
            <p>Powered by Llama 3.1 · RAG · Live Web Search</p>
        </div>
    """, unsafe_allow_html=True)

    # render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            st.markdown(msg["content"])

    # input (Streamlit pins this to bottom automatically)
    if prompt := st.chat_input("Ask a question, request a mock interview, or get resume feedback…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            # --- RAG first ---
            retrieved_chunks = retrieve(prompt, k=6)

            if retrieved_chunks:
                context = "\n\n".join(retrieved_chunks)
                search_used = False
            else:
                # --- Web search fallback ---
                with st.spinner("🌐 Searching the web…"):
                    context = search_web(prompt)
                search_used = True

            with st.spinner("AI is thinking…"):
                structured_prompt = f"""You are an expert AI Interview Coach with deep knowledge across technical and behavioural interviews.
Use ONLY the context below to answer. If the context has gaps, fill them using your expertise while staying strictly relevant to interview preparation.

Context:
{context}

Candidate's question:
{prompt}"""
                response = generate_response(structured_prompt, mode=st.session_state.detail_mode)
                st.markdown(response)

            # show context expander
            label = "🌐 Web Search Results" if search_used else "📄 Document Excerpts"
            with st.expander(label, expanded=False):
                if search_used:
                    st.markdown(context)
                else:
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        st.markdown(f"**Excerpt {i}**\n\n{chunk}")

        st.session_state.messages.append({"role": "assistant", "content": response})

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        st.error("⚠️ API key not configured. Set GROQ_API_KEY in config/config.py.")
        st.stop()

    init_session()
    st.markdown(CSS, unsafe_allow_html=True)

    chat_col, side_col = st.columns([7, 3], gap="large")
    with side_col:
        render_sidebar()
    with chat_col:
        render_chat_and_input()

if __name__ == "__main__":
    main()