import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, use_local_embeddings=False):
    """Create vector store with optional fallback to local embeddings"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Try local embeddings first if requested or if Google API fails
    if use_local_embeddings:
        try:
            status_text.text("🔄 Using LOCAL embeddings (free, no API needed)...")
            progress_bar.progress(0.1)
            
            # Use HuggingFace local embeddings (free, no API key needed)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            status_text.text(f"📊 Creating embeddings for {len(text_chunks)} chunks...")
            progress_bar.progress(0.3)
            
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            
            progress_bar.progress(0.9)
            vector_store.save_local("faiss_index")
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            st.success("✅ PDF processed successfully using local embeddings!")
            return
            
        except Exception as local_error:
            st.error(f"Error with local embeddings: {str(local_error)}")
            st.info("Falling back to Google API embeddings...")
    
    # Try Google API embeddings
    load_dotenv(override=True)
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not google_api_key:
        st.warning("⚠️ No Google API key found. Switching to LOCAL embeddings (free)...")
        return get_vector_store(text_chunks, use_local_embeddings=True)
    
    max_retries = 2
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            status_text.text(f"🔄 Using Google API embeddings... (Attempt {attempt+1}/{max_retries})")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            
            status_text.text(f"📊 Processing {len(text_chunks)} text chunks...")
            progress_bar.progress(0.5)
            
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            st.success("✅ PDF processed successfully!")
            return
            
        except Exception as e:
            error_msg = str(e)
            
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    status_text.text(f"⚠️ Quota exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    # Quota exhausted - offer local embeddings as fallback
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.warning("""
                    ⚠️ **Google API Quota Exhausted**
                    
                    Your Google Embedding API quota has been exhausted, but you can still use the app!
                    """)
                    
                    if st.button("🆓 Use FREE Local Embeddings (No API Needed)", type="primary"):
                        return get_vector_store(text_chunks, use_local_embeddings=True)
                    else:
                        st.info("💡 Tip: Click the button above to use free local embeddings with no quota limits!")
                        raise
            
            elif "PERMISSION_DENIED" in error_msg or "suspended" in error_msg.lower():
                st.warning("⚠️ Google API key issue. Switching to local embeddings...")
                return get_vector_store(text_chunks, use_local_embeddings=True)
            
            else:
                st.error(f"Error: {error_msg}")
                if st.button("Try Local Embeddings Instead"):
                    return get_vector_store(text_chunks, use_local_embeddings=True)
                raise


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    load_dotenv(override=True)  # Reload in case env vars changed
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        return None
    
    # Try different model names in order of preference
    # Updated to use models that are actually available
    model_names = [
        "gemini-2.5-flash",    # Latest fast model (most likely to work)
        "gemini-2.0-flash",     # Fast model
        "gemini-flash-latest",  # Latest flash model
        "gemini-pro-latest",    # Latest pro model
        "gemini-1.5-flash",    # Older but still available
        "gemini-1.5-pro",      # Older but still available
        "gemini-1.0-pro",      # Legacy model
    ]
    
    model = None
    last_error = None
    
    for model_name in model_names:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                google_api_key=google_api_key
            )
            # Test if model works by creating it (will fail fast if not available)
            break
        except Exception as e:
            last_error = e
            continue
    
    if model is None:
        st.error(f"""
        ⚠️ **Unable to Access Gemini Models**
        
        Your API key doesn't have access to any of the tested Gemini models.
        
        **Error:** {str(last_error)}
        
        **Possible Solutions:**
        1. Enable Gemini API in Google Cloud Console
        2. Check if your API key has proper permissions
        3. Try generating a new API key
        4. Verify billing is enabled (required for some models)
        
        **Note:** You can still use local embeddings for PDF processing,
        but chat functionality requires a working Gemini API access.
        """)
        return None

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question, use_local_embeddings=False):
    try:
        # Try to load with the same embedding model that was used to create it
        # First try local embeddings, then Google API
        try:
            if use_local_embeddings:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            else:
                load_dotenv(override=True)
                google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not google_api_key:
                    # Fallback to local embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                else:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question, k=3)
            
            # Show found documents even if chat doesn't work
            if not docs:
                st.warning("No relevant content found in the PDF for your question.")
                return
                
        except Exception as load_error:
            # Try the other embedding type
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question, k=3)
            except:
                st.error("Please upload and process PDF files first before asking questions.")
                return
    except Exception as e:
        st.error("Please upload and process PDF files first before asking questions.")
        return

    # Try to get chain, but if it fails, at least show relevant content
    chain = get_conversational_chain()
    
    if chain is None:
        # If chat doesn't work, at least show the relevant documents
        st.warning("⚠️ **Chat unavailable** - Gemini API models not accessible, but here are relevant sections from your PDF:")
        st.markdown("---")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Relevant Section {i}:**")
            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            st.markdown("---")
        st.info("💡 **Tip:** Enable Gemini API to get AI-generated answers instead of just document search.")
        return
    
    try:
        with st.spinner("Generating answer..."):
            response = chain(
                {"input_documents":docs, "question": user_question}
                , return_only_outputs=True)

            print(response)
            st.write("**Reply:**", response["output_text"])
    except Exception as e:
        error_msg = str(e)
        if "NOT_FOUND" in error_msg or "404" in error_msg:
            # Still show relevant docs even if chat fails
            st.warning("⚠️ **Chat unavailable** - Here are relevant sections from your PDF:")
            st.markdown("---")
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Relevant Section {i}:**")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.markdown("---")
            st.error(f"""
            ⚠️ **Model Not Found Error**
            
            **Error:** {error_msg}
            
            **To fix this:**
            1. Enable Gemini API: https://console.cloud.google.com/apis/library (search for "Generative Language API")
            2. Generate new API key: https://makersuite.google.com/app/apikey
            """)
        else:
            st.error(f"Error generating response: {error_msg}")




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    # Store embedding preference in session state
    if 'use_local_embeddings' not in st.session_state:
        st.session_state.use_local_embeddings = True  # Default to local (free)

    if user_question:
        user_input(user_question, use_local_embeddings=st.session_state.use_local_embeddings)

    with st.sidebar:
        st.title("Menu:")
        
        # Embedding method selection
        embedding_method = st.radio(
            "Choose Embedding Method:",
            ["🆓 Local (Free, No API)", "🌐 Google API (Requires API Key)"],
            help="Local embeddings are free with no quota limits. Google API has quota restrictions.",
            index=0 if st.session_state.get('use_local_embeddings', True) else 1
        )
        use_local = embedding_method.startswith("🆓")
        st.session_state.use_local_embeddings = use_local
        
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, use_local_embeddings=use_local)
        
        st.markdown("---")
        st.markdown("### 💡 Tips:")
        st.info("""
        **Free Option:** Use "Local (Free, No API)" to avoid quota limits!
        
        Works offline, no API key needed, unlimited usage.
        """)



if __name__ == "__main__":
    main()