import streamlit as st
from qa import QAPipeline
from ingest import process_documents
import os
import shutil
import random

def reset_database():
    """Clear all stored data and reset session state"""
    try:
        if 'qa_pipeline' in st.session_state:
            st.session_state.qa_pipeline.retriever.vector_store.delete_collection()
        
        if os.path.exists("documents"):
            shutil.rmtree("documents")
            os.makedirs("documents")
        
        st.session_state.qa_pipeline = QAPipeline()
        st.success("Database cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

def main():
    st.set_page_config(page_title="DocuMind AI", layout="wide")
    
    # Initialize session state
    if 'qa_pipeline' not in st.session_state:
        st.session_state.qa_pipeline = QAPipeline()
    if 'ingested' not in st.session_state:
        st.session_state.ingested = False
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"],
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ingest Documents"):
                if uploaded_files:
                    try:
                        reset_database()
                        os.makedirs("documents", exist_ok=True)
                        for file in uploaded_files:
                            with open(os.path.join("documents", file.name), "wb") as f:
                                f.write(file.getbuffer())
                        with st.spinner("Processing documents..."):
                            process_documents()
                            st.session_state.ingested = True
                            st.success(f"Ingested {len(uploaded_files)} documents!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                else:
                    st.warning("Please upload documents first")
        with col2:
            st.button("Clear Database", on_click=reset_database)

    # Main interface
    st.title("📄 Document AI Assistant")
    
    # Display chat history with Streamlit's default styling
    for qa in st.session_state.history:
        with st.expander(f"Q: {qa['question']}", expanded=False):
            # Thinking Process Section
            with st.container():
                st.markdown("#### 🧠 Thinking Process of the Assistant")
                st.markdown(qa['thinking'])
                st.markdown("---")
            
            # Final Answer Section
            with st.container():
                st.markdown("#### 📝 Final Answer")
                st.markdown(qa['answer'])
                st.markdown("---")
            
            # Source Documents Section
            if qa["sources"]:
                st.markdown("#### 🔍 Source Documents")
                for idx, source in enumerate(qa["sources"], 1):
                    with st.container():
                        st.write(f"📄 **{source['source']}** (Page {source['page']})")
                        st.text_area(
                            "Relevant text excerpt:",
                            value=source['full_text'],
                            key=f"source_{idx}_{random.randint(0, 999999)}",
                            disabled=True
                        )

    # Question input with processing spinner
    question = st.chat_input(
        "Ask a question about your documents:",
        disabled=not st.session_state.ingested
    )

    # Process question with visual feedback
    if question and st.session_state.ingested:
        # Add to history immediately
        st.session_state.history.append({
            "question": question,
            "thinking": "",
            "answer": "Processing...",
            "sources": []
        })
        
        try:
            with st.status("🧠 Processing your question...", expanded=True) as status:
                st.write("🔍 Retrieving relevant documents...")
                result = st.session_state.qa_pipeline.generate_answer(question)
                status.update(label="✅ Processing complete", state="complete")
            
            # Update history with results
            st.session_state.history[-1] = {
                "question": question,
                "thinking": result.get('thinking', ''),
                "answer": result.get('answer', 'No answer generated'),
                "sources": result.get('sources', [])
            }
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.session_state.history.pop()

if __name__ == "__main__":
    main()