from dotenv import load_dotenv
load_dotenv(dotenv_path=r"\.env", override=True)


from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import argparse

import os, shutil
from openai import OpenAI

DATA_PATH = "data/"
CHROMA_PATH = "chroma"


def create_database():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the data first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    # create new DB from the documents
    db = Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=CHROMA_PATH
    )
    print(f"Save {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    create_database()

    