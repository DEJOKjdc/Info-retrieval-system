import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation.invoke({
            'question': user_question
        })
        st.session_state.chatHistory = response["chat_history"]

        for i, message in enumerate(st.session_state.chatHistory):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)


def main():
    """
    Main function for the Streamlit application with a more stylish layout.
    """
    # Set a more stylish page configuration
    st.set_page_config(
        page_title="Information Retrieval System",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.header("Information Retrieval System 📚")

    # Initialize session state variables with a more concise syntax
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Display the chat history from st.session_state.chatHistory
    if st.session_state.chatHistory:
        for message in st.session_state.chatHistory:
            role = "user" if message.type == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    # Use st.chat_input for a more integrated text input experience
    user_question = st.chat_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        st.markdown("Upload your PDF files and process them to start asking questions.")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Submit & Process", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing... This might take a few moments."):
                    # Call helper functions to process PDFs and set up the conversation chain
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    
                    # Store the vector store and conversational chain in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    
                    st.success("Processing complete! You can now ask questions.")
                    st.session_state.chatHistory = [] # Clear chat history for new documents
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
