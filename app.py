import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────────
# SIMPLE DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

/* Hide streamlit header */
header {
    visibility: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Buttons */
.stButton button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #111827;
    color: #ffffff !important;
}

/* Text input area */
textarea {
    color: black!important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    padding: 1rem;
    border-radius: 12px;
}

/* Messages */
.user-msg {
    background-color: #2563eb;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.7rem;
    color: white;
}

.ai-msg {
    background-color: #1e293b;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}

.title {
    font-size: 3rem;
    font-weight: 700;
    color: white;
}

.subtitle {
    font-size: 1rem;
    color: #cbd5e1;
    margin-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────
os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ─────────────────────────────────────────────
# BUILD RETRIEVER
# ─────────────────────────────────────────────
def build_retriever(pdf_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever()

    return retriever


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def get_llm():
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.4,
        mistral_api_key=st.secrets["MISTRAL_API_KEY"]
    )


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a document assistant.

        Answer ONLY from the provided context.

        If answer is not found, say:
        'I could not find the answer in the document.'
        """
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


# ─────────────────────────────────────────────
# ASK FUNCTION
# ─────────────────────────────────────────────
def ask_question(query):

    docs = st.session_state.retriever.invoke(query)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    final_prompt = PROMPT.invoke({
        "context": context,
        "question": query
    })

    response = get_llm().invoke(final_prompt)

    return response.content


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.title("📄 DocMind")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("Process Document"):

        if uploaded_file is None:
            st.error("Please upload a PDF.")

        else:
            with st.spinner("Processing..."):

                retriever = build_retriever(
                    uploaded_file.read()
                )

                st.session_state.retriever = retriever
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []

                st.success("Document ready.")

    st.divider()

    if st.session_state.retriever:
        st.success(f"Loaded: {st.session_state.pdf_name}")
    else:
        st.info("No document loaded")


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown(
    '<div class="title">DocMind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about your PDF</div>',
    unsafe_allow_html=True
)


# CHAT HISTORY
for chat in st.session_state.chat_history:

    st.markdown(
        f"""
        <div class="user-msg">
        <b>You:</b><br>
        {chat["question"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="ai-msg">
        <b>DocMind:</b><br>
        {chat["answer"]}
        </div>
        """,
        unsafe_allow_html=True
    )


# CHAT INPUT
query = st.chat_input(
    "Ask a question about your PDF..."
)


# ASK
if query:

    if st.session_state.retriever is None:

        st.warning("Please upload and process a PDF first.")

    else:

        with st.spinner("Thinking..."):

            answer = ask_question(query)

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer
        })

        st.rerun()