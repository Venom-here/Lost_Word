## When using Ollama Embedding, it takes a lot of time to embed the entire text and hence requires a lot of time to run on streamlit

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## 
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate missing word based on the question.
    <context>
    {context}
    <context>
    Question:{input}

    """
)


def create_vector_embedding():
    if "vectors" not in st.session_state: ## session_state helps to remember vectorStore DB

        # st.session_state.embeddings=(OllamaEmbeddings(model="gemma2:2b"))
        st.session_state.embeddings=NVIDIAEmbeddings()

        # st.session_state.loader=PyPDFLoader('data.pdf')
        st.session_state.loader=TextLoader("data.txt") ## Data Ingestion step

        st.session_state.docs=st.session_state.loader.load() ## Document Loading

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=15,chunk_overlap=5)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) # to load entire pdf, remove -> [:50]
        
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
st.title("Lost Word Predictor")


user_prompt = st.text_input("Enter your sentence from the data")


if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")


import time


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt) ## Creates a chain for passing a list of documents to a model.
    ## it sends all the list of documents to the {context} (under prompt)
    
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)


    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt}) # {input} --> prompt
    print(f"Response time :{time.process_time()-start}")


    st.write(response['answer'])
