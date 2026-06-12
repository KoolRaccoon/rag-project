from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\CodingProjects\rag_project\.env", override=True)

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from groq import Groq
import argparse
from create_database import create_database

import os, shutil

from openai import OpenAI

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

DATA_PATH = "data/"
CHROMA_PATH = "chroma"


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_function)


    #search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.2:
        print (f"Unable to find matching results")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question = query_text)
    
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}])
    
    print(response.choices[0].message.content)

    #model = ChatHuggingFace()
    

if __name__ == "__main__":
    if not os.path.exists(CHROMA_PATH):
        create_database()
    main()
