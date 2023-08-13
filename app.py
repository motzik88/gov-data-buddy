import openai
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import streamlit as st
from collections import deque
from data.fewshotsexamples import *

# load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
OPENAI_EMBEDDING_DEPLOYMENT_ENDPOINT = os.getenv('OPENAI_EMBEDDING_DEPLOYMENT_ENDPOINT')
OPENAI_EMBEDDING_API_KEY = os.getenv('OPENAI_EMBEDDING_API_KEY')

# init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY


prompt_template = FEW_SHOTS_EXAMPLES



PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}



# init openai
llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                              openai_api_key=OPENAI_EMBEDDING_API_KEY, openai_api_base=OPENAI_EMBEDDING_DEPLOYMENT_ENDPOINT,
                              chunk_size=1)

# load the faiss vector store we saved into memory
vectorStore = FAISS.load_local(r"./dbs/documentation/faiss_index", embeddings)

# use the faiss vector store we saved to search the local document
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# use the vector store as a retriever
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)


 
 
def ask_question(question, print_source_documents=True, print_answer=False):
    result = qa({"query": question})
    if print_answer:
        print("Question:", question)
        print("Answer:", result["result"])
        if print_source_documents:
            print("Source documents:")
            for doc in result["source_documents"]:
                print(doc)
    else:
        return result["result"], result["source_documents"]


def get_answer_and_context(question):
    answer, context = ask_question(question)
    return answer, context


style = "text-align: right; unicode-bidi:embed; direction: RTL;"


# Main app function
def main():
    st.markdown("""
    <style>
    input {
      unicode-bidi:embed;
      direction: RTL;
    }
    </style>
        """, unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    st.markdown("<h1 style='text-align: center; color: blue;'>סיפור התקציב</h1>", unsafe_allow_html=True)
    
    # Input question
    question = st.text_input(".הזינו שאלה על ספר התקציב. נסו לשאול על נושאים כמו תקציב משרד החינוך, תקציב משרד הבריאות, או כל דבר אחר על ספר התקציב")

    # Get answer and context from the backend
    if question:
        answer, context = get_answer_and_context(question)
        current = (question, answer, context)
        st.session_state['chat_history'].append(current)
        
        
        # Display chat history
        q, a, c = current
        # st.markdown(f'<div style="{style}">  <b> שאלה: </b> {q} </div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(f'<div style="{style}"> <b> תשובה: </b> {a} </div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
    
        # Display context in an expandable container
        with st.expander("**.לחצו כאן על מנת לצפות במקורות המידע הרלוונטיים עבור השאלה שלכם. בנוסף לטקסט, יופיע מקור הטקסט במסמך**"):
            for i, ctx in enumerate(c):
                st.markdown(f"<h4 style='{style}'>מקור מידע {i+1}:</h4> <br>", unsafe_allow_html=True)
                st.markdown(f"<h6 style='{style}'>טקסט:</h6> <br>", unsafe_allow_html=True)
                st.markdown(f"<div style='{style}'>{ctx.page_content}</div> <br>", unsafe_allow_html=True)
                st.markdown(f"<h6 style='{style}'>מקור:</h6> <br>", unsafe_allow_html=True)
                # st.markdown(f"<div> {ctx.metadata}</div> <br>", unsafe_allow_html=True)
                for key, value in ctx.metadata.items():
                    st.markdown(f"<div>{key}: {value}</div>", unsafe_allow_html=True)
        st.write("---")
        
        if len(st.session_state['chat_history']) > 1:
            with st.sidebar:
                st.markdown(f'<h3 style="{style}"> היסטוריית שיחה </h3>', unsafe_allow_html=True)
                for q, a, c in st.session_state['chat_history'][-5:-1]:
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.markdown(f'<div style="{style}"> <b> שאלה: </b> {q} </div>', unsafe_allow_html=True)
                    # st.markdown('<br>', unsafe_allow_html=True)
                    st.markdown(f'<div style="{style}"> <b> תשובה: </b> {a} </div>', unsafe_allow_html=True)
                    st.markdown('<hr>', unsafe_allow_html=True)
                    

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     while True:
#         query = input('you: ')
#         if query == 'q':
#             break
#         ask_question(query, print_answer=True)
