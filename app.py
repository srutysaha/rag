import streamlit as st
from vectorstore import crawl_website, build_vector_store
from rag_chain import build_rag_chain

st.set_page_config(page_title="🌍 Website RAG Chatbot", layout="wide")
st.title("💬 Ask Questions About Any Website")

url = st.text_input("Enter a website URL", placeholder="https://example.com")

if st.button("🌐 Crawl Website & Build Knowledge Base"):
    if url:
        with st.spinner("Crawling and vectorizing..."):
            pages = crawl_website(url)
            build_vector_store(pages)
        st.success("✅ Website crawled and stored!")
    else:
        st.error("Please enter a valid URL.")

st.markdown("---")
st.subheader("Ask a Question")

query = st.text_input("Your question:")

if st.button("🧠 Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            chain = build_rag_chain()
            result = chain(query)
            st.success("✅ Answer:")
            st.write(result['result'])

            # with st.expander("📄 Source Documents"):
            #     for doc in result['source_documents']:
            #         st.markdown(doc.page_content[:300] + "...")
    else:
        st.warning("Please enter a question.")
