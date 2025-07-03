from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline

def load_flan_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3
    )

    return HuggingFacePipeline(pipeline=pipe)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_rag_chain(persist_dir="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that answers only from the provided website content.
If the answer is not present, say: "I couldn't find that in the website content."

Website Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = load_flan_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        # return_source_documents=True
    )
