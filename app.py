import os
import re

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: #fafafa;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-top: 0.5rem;
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# INITIALIZE GROQ CLIENT
# ---------------------------------------------------------

def get_groq_client():
    api_key = None

    # Streamlit Cloud secrets
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

    # Local environment variable fallback
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    text = ""

    pdf_bytes = pdf_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text")

        if page_text.strip():
            text += f"\n[Page {page_number}]\n"
            text += page_text

    document.close()

    return text


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def create_chunks(text):
    """
    Creates overlapping chunks while attempting to preserve
    paragraph boundaries.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding the paragraph stays within the target size
        if len(current_chunk) + len(paragraph) + 1 <= CHUNK_SIZE:
            current_chunk += paragraph + "\n"

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Create overlap from the end of the previous chunk
            overlap_text = current_chunk[-CHUNK_OVERLAP:]

            current_chunk = overlap_text + "\n" + paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------
# CREATE FAISS VECTOR DATABASE
# ---------------------------------------------------------

def create_faiss_index(chunks, model):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings


# ---------------------------------------------------------
# SEARCH RELEVANT CHUNKS
# ---------------------------------------------------------

def search_documents(question, chunks, index, model, top_k=TOP_K):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        results.append(
            {
                "chunk": chunks[index_position],
                "score": float(score),
                "index": int(index_position)
            }
        )

    return results


# ---------------------------------------------------------
# GENERATE ANSWER WITH GROQ
# ---------------------------------------------------------

def generate_answer(question, search_results, client):

    context_parts = []

    for i, result in enumerate(search_results, start=1):
        context_parts.append(
            f"--- Policy Section {i} ---\n"
            f"{result['chunk']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using ONLY the HR policy
information provided in the retrieved context.

Rules:

1. Do not invent HR policies.
2. Do not use outside information as if it came from the policy.
3. If the answer cannot be found in the provided context,
   clearly say that the uploaded policy does not provide
   enough information.
4. Give concise, professional and easy-to-understand answers.
5. When useful, mention the relevant policy section or page.
6. If the policy contains conditions, exceptions or limits,
   include them.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question based strictly on the HR policy context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📘 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an HR policy PDF and ask questions using Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("**Embedding Model**")
    st.code(EMBEDDING_MODEL)

    st.write("**LLM**")
    st.code(GROQ_MODEL)

    st.write("**Vector Database**")
    st.code("FAISS")

    st.divider()

    st.write(
        "Upload a policy document, process it, "
        "and then ask questions about its contents."
    )


# ---------------------------------------------------------
# API KEY CHECK
# ---------------------------------------------------------

groq_client = get_groq_client()

if groq_client is None:

    st.warning(
        "⚠️ Groq API key is not configured. "
        "Add GROQ_API_KEY to Streamlit Secrets before asking questions."
    )


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF."
)


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    if (
        st.session_state.document_name != uploaded_file.name
        or not st.session_state.processed
    ):

        if st.button("🔎 Process Policy Document", type="primary"):

            with st.spinner("Processing HR policy..."):

                try:

                    # Step 1: Extract text
                    raw_text = extract_text_from_pdf(uploaded_file)

                    if not raw_text.strip():
                        st.error(
                            "No readable text was found in this PDF. "
                            "If it is a scanned PDF, OCR may be required."
                        )
                        st.stop()

                    # Step 2: Clean text
                    cleaned_text = clean_text(raw_text)

                    # Step 3: Create chunks
                    chunks = create_chunks(cleaned_text)

                    if not chunks:
                        st.error("Could not create text chunks from the PDF.")
                        st.stop()

                    # Step 4: Load embedding model
                    embedding_model = load_embedding_model()

                    # Step 5: Create FAISS index
                    index, _ = create_faiss_index(
                        chunks,
                        embedding_model
                    )

                    # Store in session state
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.processed = True

                    st.success("✅ HR policy processed successfully!")

                except Exception as e:

                    st.error(
                        f"An error occurred while processing the PDF: {e}"
                    )


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if st.session_state.processed:

    st.success(
        f"📄 **{st.session_state.document_name}** is ready for questions."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )

    with col2:
        st.metric(
            "Retrieved Chunks",
            TOP_K
        )


# ---------------------------------------------------------
# QUESTION SECTION
# ---------------------------------------------------------

st.divider()

st.subheader("💬 Ask about the HR Policy")

question = st.text_input(
    "Enter your question",
    placeholder="Example: How many annual leave days are employees entitled to?"
)


if st.button("Ask Question", type="primary"):

    if not uploaded_file:
        st.warning("Please upload an HR policy PDF first.")

    elif not st.session_state.processed:
        st.warning("Please process the HR policy document first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    elif groq_client is None:
        st.error(
            "Groq API key is missing. Configure GROQ_API_KEY in "
            "Streamlit Secrets."
        )

    else:

        with st.spinner("Searching the policy and generating an answer..."):

            try:

                embedding_model = load_embedding_model()

                # RAG retrieval
                search_results = search_documents(
                    question,
                    st.session_state.chunks,
                    st.session_state.index,
                    embedding_model,
                    TOP_K
                )

                # Generate answer
                answer = generate_answer(
                    question,
                    search_results,
                    groq_client
                )

                # Display answer
                st.subheader("🤖 Answer")

                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )

                # Display retrieved sources
                st.subheader("📚 Retrieved Policy Sections")

                for i, result in enumerate(search_results, start=1):

                    with st.expander(
                        f"Policy Section {i} — Similarity: "
                        f"{result['score']:.3f}"
                    ):

                        st.write(result["chunk"])

            except Exception as e:

                st.error(
                    f"An error occurred while generating the answer: {e}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "HR Policy Assistant • RAG • FAISS • Sentence Transformers • "
    "PyMuPDF • Groq"
)
