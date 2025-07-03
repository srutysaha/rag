import os
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def crawl_website(base_url, max_pages=50):
    visited = set()
    to_visit = [base_url]
    pages = []

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=5)
            if 'text/html' not in resp.headers.get('Content-Type', ''):
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator="\n", strip=True)
            pages.append({'url': url, 'text': text})

            for link in soup.find_all('a', href=True):
                full_url = requests.compat.urljoin(url, link['href'])
                if base_url in full_url and full_url not in visited:
                    to_visit.append(full_url)
        except Exception:
            continue

    return pages

def build_vector_store(pages, persist_dir="db"):
    raw_text = "\n\n".join([p['text'] for p in pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb
