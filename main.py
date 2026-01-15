import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os


def get_pdf_text(pdf_docs):
   text = ""
   for pdf in pdf_docs:
      pdf_reader = PdfReader(pdf)
      for page in pdf_reader.pages:
         text += page.extract_text()
   return text



def get_text_chunks(text):
   text_splitter = RecursiveCharacterTextSplitter( separators="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
   chunks = text_splitter.split_text(text)

   return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Use Azure OpenAI with DeepSeek model
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
   

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Title with better styling
    st.title("💬 Chat with PDF")
    st.markdown("Ask me anything about your uploaded documents!")
    
    # Clear chat button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button(" Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Display chat history
    if not st.session_state.messages:
        st.info("Upload a PDF from the sidebar and start chatting!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💭 Ask a question about your document..."):
        if st.session_state.conversation:
            # Display user message
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response
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

    # Sidebar
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
                    # Get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Split pdf text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vector_store = get_vectorstore(text_chunks)
                    
                    # Create conversation chain
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




if __name__ == "__main__":
    main()