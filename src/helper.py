import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import asyncio
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
import streamlit as st



load_dotenv()  # Loads .env for local development

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY") or
    st.secrets.get("google_genai", {}).get("api_key")
)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    st.error("Google API Key not found in .env or Streamlit secrets.")


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits a given text into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates and returns a FAISS vector store from text chunks.
    This function now handles the `asyncio` event loop error.
    """
    # Check if there is a running event loop and create one if not.
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",         # Important
        output_key="answer",          # Match what LangChain expects
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=False,
        output_key="answer"
    )
    return chain
