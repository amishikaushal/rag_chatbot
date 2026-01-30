import streamlit as st
import os
from dotenv import load_dotenv
from src.rag import get_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------
# 1. App Setup
# --------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="AI Research Lab",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# 2. Helper: Detect small talk
# --------------------------------------------------
def is_small_talk(query: str) -> bool:
    greetings = {
        "hi", "hello", "hey", "how are you", "how r you",
        "good morning", "good evening", "good afternoon"
    }
    q = query.lower().strip()
    return q in greetings or len(q.split()) <= 3

# --------------------------------------------------
# 3. Initialize Engines & Memory
# --------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain()

if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# 4. Premium CSS
# --------------------------------------------------
st.markdown("""
<style>
:root { --bg:#F7F9FB; --accent:#0f172a; --muted:#6B7280; --card:#ffffff; --user:#0b84ff; --assistant:#f2f6ff; --border:#E6EAF0; --radius:14px;}
html, body, [class*="css"]{ font-family:'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; background:var(--bg) !important; color:#0f172a;}
.block-container{max-width:920px; padding-top:2.5rem; padding-bottom:3rem;}
.header {
  display:flex; align-items:center; gap:12px; padding:18px 24px; border-radius:12px;
  background:linear-gradient(90deg, rgba(15,23,42,1) 0%, rgba(12,74,255,1) 100%);
  color:white; box-shadow: 0 6px 18px rgba(12,74,255,0.08); margin-bottom:18px;
}
.header .logo {font-size:22px; font-weight:700; display:flex; align-items:center; gap:10px;}
.header .subtitle{font-size:12px; opacity:0.9;}
.chat-container{background:linear-gradient(180deg, rgba(255,255,255,0.8), rgba(255,255,255,0.6)); padding:18px; border-radius:14px; box-shadow: 0 4px 20px rgba(16,24,40,0.06); border:1px solid var(--border);}
.stChatMessage { padding: 12px 0; display:flex; flex-direction:column; gap:8px; }
.stChatMessage.user { align-items:flex-end; }
.stChatMessage.user p {
  background: linear-gradient(90deg, var(--user), #0057d9); color: white; padding: 10px 14px; border-radius:14px 14px 4px 14px; max-width:75%; display:inline-block; box-shadow: 0 6px 18px rgba(11,132,255,0.12);
}
.stChatMessage.assistant { align-items:flex-start; }
.stChatMessage.assistant p {
  background: var(--card); color: #0f172a; padding: 12px 14px; border-radius:14px 14px 14px 4px; max-width:85%; display:inline-block; border:1px solid var(--border); box-shadow: 0 6px 18px rgba(2,6,23,0.04);
}
.meta { font-size:12px; color:var(--muted); margin-top:4px; }
.chat-footer { display:flex; gap:10px; align-items:center; margin-top:12px; }
input[data-testid="stTextArea"]{ border-radius:12px !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#061025,#021028); color:#E6EEF8; padding:18px;}
[data-testid="stSidebar"] .stButton>button{ border-radius:10px; background: linear-gradient(90deg,#0066ff,#0047b3); border:none;}
footer, header { visibility: hidden; }
</style>

<div class="header"><div class="logo">🧠 AI Research Lab</div><div class="subtitle">Private Knowledge Assistant</div></div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 AI Research Lab")
    st.caption("Private Knowledge Assistant")
    st.markdown("---")
    st.markdown("- Hybrid RAG (PDF + LLM)")
    st.markdown("- Academic-grade answers")
    st.markdown("- Clean citations")

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# 6. Render Chat History
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# 7. Chat Input
# --------------------------------------------------
if prompt := st.chat_input("Ask a question about AI / ML / research papers…"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(""):

            # ✅ Small talk → pure LLM
            if is_small_talk(prompt):
                answer = st.session_state.llm.invoke(prompt).content
                sources = []

            # ✅ Knowledge → RAG
            else:
                response = st.session_state.rag_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = list(set(
                    os.path.basename(doc.metadata.get("source", ""))
                    for doc in response.get("source_documents", [])
                    if doc.metadata.get("source")
                ))

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })