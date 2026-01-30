from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_PATH = "vectorstore"

def get_rag_chain():
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load Vector Store
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Vector store directory '{DB_PATH}' not found. Run ingest.py first."
        )

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )

    # 4. CLEAN HYBRID PROMPT (✅ no notes, no noise)
    hybrid_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant.

Use the provided context from uploaded PDFs if it is relevant to the question.
If the context does NOT contain the answer, answer naturally using your general knowledge.
Do NOT mention documents unless they are actually used.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 5. RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": hybrid_prompt},
        return_source_documents=True
    )

    return qa_chain