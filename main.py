# Importing necesary libraries
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#take environment variables from .env.
from dotenv import load_dotenv
load_dotenv() 

#setting up the basic UI
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_store.pkl"
main_placeholder = st.empty() #empty placeholder to display the progress bar
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

if process_url_clicked: # If this is True or if the button is clicked
    # Load the QA with sources chain
    loaders = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Loading the data from the URLs!!!!")
    data=loaders.load()
    #Split the text into paragraphs or chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text from the URLs into chunks!!!!")
    docs = text_splitter.split_documents(data)
    #Create embeddings and save it to the FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    #embeddings_data = vectorstore_openai.embeddings.serialize()
    main_placeholder.text("Embedding Vector started building!!!!")
    time.sleep(2)

    #Save the vector store to a pkl file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai,f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)