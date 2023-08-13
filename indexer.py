from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import openai
import os
import form_recognizer


# load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")


# init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                                  chunk_size=1)
    dataPath = "./data/documentation/"
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    all_pages = []
    for file in os.listdir(dataPath):
        if file.endswith(".pdf"):
            fileName = dataPath + file
            # read and extract the text from the pdf, including chunking
            pages = form_recognizer.analyze_read(fileName)
            all_pages.extend(pages)

    # Use Langchain to create the embeddings using text-embedding-ada-002
    db = FAISS.from_documents(documents=all_pages, embedding=embeddings)

    # save the embeddings into FAISS vector store
    db.save_local("./dbs/documentation/faiss_index")
    
    
