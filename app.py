import os
import streamlit as st
import time
from dotenv import load_dotenv
import hashlib

# LangChain imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Check API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Streamlit UI setup
st.title("📰News Research Tool")
st.sidebar.title("🔗 Enter up to 3 News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url)

process_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Setup Gemini LLM + Embeddings
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Function to hash URLs to a session key
def get_session_key(urls):
    return hashlib.md5("".join(urls).encode()).hexdigest()

# Process URLs and create FAISS index
if process_clicked and urls:
    try:
        with st.spinner("📥 Loading content from URLs..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

        if not data:
            st.warning("⚠️ No content could be loaded from the URLs.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=500,
                chunk_overlap=50
            )

            docs = text_splitter.split_documents(data)

            if not docs:
                st.warning("⚠️ No chunks were generated from the documents.")
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)
                session_key = get_session_key(urls)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["session_key"] = session_key
                st.success("✅ URLs processed and content indexed.")

                with st.expander("📝 Preview extracted content"):
                    for doc in docs[:3]:
                        st.write(doc.page_content[:500])

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

# Ask a question
query = main_placeholder.text_input("Ask a question about the articles:")

# Answer the question
if query:
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please process URLs first.")
    else:
        try:
            retriever = st.session_state["vectorstore"].as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)

            st.header("🧠 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("🔗 Sources")
                for source in sources.split("\n"):
                    if source.strip():
                        st.write(source)

        except Exception as e:
            st.error(f"❌ Error answering the question: {e}")
