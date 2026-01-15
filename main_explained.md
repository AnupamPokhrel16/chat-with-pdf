# main.py Line-by-Line Explanation

Below is a detailed explanation of each line and block in your `main.py` file for the Streamlit PDF chat app.

---

```python
import streamlit as st
```
- Imports Streamlit, the web app framework used for the UI.

```python
from dotenv import load_dotenv
```
- Imports the function to load environment variables from a `.env` file.

```python
from PyPDF2 import PdfReader
```
- Imports the PDF reader to extract text from PDF files.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
- Imports a text splitter to break large text into manageable chunks.

```python
from langchain_huggingface import HuggingFaceEmbeddings
```
- Imports HuggingFace embeddings for converting text to vectors.

```python
from langchain_community.vectorstores import FAISS
```
- Imports FAISS, a vector database for efficient similarity search.

```python
from langchain_Cclassic.chains.conversational_retrieval.base import ConversationalRetrievalChain
```
- Imports a conversational retrieval chain for question-answering over documents.

```python
from langchain_openai import ChatOpenAI
```
- Imports the OpenAI chat model interface.

```python
import os
```
- Imports the OS module for environment variable access.

---

### PDF Text Extraction

```python
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```
- Reads all uploaded PDFs and extracts their text content.

---

### Text Chunking

```python
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks
```
- Splits the extracted text into overlapping chunks for better retrieval.

---

### Vector Store Creation

```python
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```
- Converts text chunks into embeddings and stores them in a FAISS vector database.

---

### Conversation Chain Setup

```python
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "DeepSeek-V3.2"),
        temperature=0.7
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return conversation_chain
```
- Sets up the conversational retrieval chain using the OpenAI model and the FAISS retriever.

---

### Main Streamlit App

```python
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")
```
- Loads environment variables and sets up the Streamlit page.

```python
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
```
- Initializes session state variables for conversation, chat history, and messages.

```python
    st.title("💬 Chat with PDF")
    st.markdown("Ask me anything about your uploaded documents!")
```
- Sets the app title and description.

```python
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button(" Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
```
- Adds a button to clear the chat and history.

```python
    st.divider()
```
- Adds a visual divider.

```python
    if not st.session_state.messages:
        st.info("Upload a PDF from the sidebar and start chatting!")
```
- Shows an info message if no chat has started.

```python
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
```
- Displays the chat history with user and assistant avatars.

```python
    if prompt := st.chat_input("💭 Ask a question about your document..."):
        if st.session_state.conversation:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Analyzing your document..."):
                    response = st.session_state.conversation({
                        'question': prompt,
                        'chat_history': st.session_state.chat_history
                    })
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    st.session_state.chat_history.append((prompt, answer))
        else:
            st.warning("Please upload and process a PDF first!")
```
- Handles user input, gets a response from the LLM, and updates the chat.

---

### Sidebar: PDF Upload and Processing

```python
    with st.sidebar:
        st.header("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDF files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        if st.button(" Process Documents", type="primary", use_container_width=True):
            if pdf_docs:
                with st.spinner(" Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("✅ Documents processed successfully! Start chatting below.")
            else:
                st.warning("Please upload at least one PDF file first.")
        st.divider()
        st.markdown("###Tips")
        st.markdown("""
        - Ask specific questions about your document
        - Request summaries or key points
        - Ask for explanations of complex topics
        - Compare information across sections
        """)
        if st.session_state.messages:
            st.divider()
            st.markdown(f"**Chat History:** {len([m for m in st.session_state.messages if m['role'] == 'user'])} questions asked")
```
- Handles PDF upload, processing, and displays usage tips and chat stats.

---

```python
if __name__ == "__main__":
    main()
```
- Runs the app if the script is executed directly.

---

This markdown file explains each line and block of your Streamlit PDF chat application. If you need even more granular explanations, let me know!
