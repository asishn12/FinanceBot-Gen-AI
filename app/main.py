import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("**:grey[👨🏻‍🔬 Finance Bot - Research Analyst]**")
st.markdown("**:grey[An Agent who can crawl and finegrain the results]**",text_alignment="justify")
st.sidebar.title(":orange[News Articles]")

query = st.chat_input("Your Question... ",max_chars = 550)


if 'history' not in st.session_state:
    st.session_state.history = []

urls = []

for i in range(3):
    url = st.sidebar.text_input(f":blue[url {i+1}]", placeholder= "enter the url")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")

st.sidebar.divider()

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
      model = "openai/gpt-oss-20b"
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)

    text1, text2, text3, text4 = "Data Loading...Started...✅✅✅", "Text Splitter...Started...✅✅✅", "Embedding Vector Started Building...✅✅✅", "Embedding Vector completed...✅✅✅"
    my_bar = main_placeholder.progress(0, text = text1)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text = text1)
    time.sleep(1)
    my_bar.empty()

    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    my_bar = main_placeholder.progress(0, text = text2)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text = text2)
    time.sleep(1)
    my_bar.empty()

    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings()

    my_bar = main_placeholder.progress(0, text = text3)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text = text3)
    time.sleep(1)
    my_bar.empty()

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    my_bar = main_placeholder.progress(0, text = text4)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text = text4)
    time.sleep(1)
    my_bar.empty()

    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


if query:
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"query": query}, return_only_outputs=True)
            st.markdown(":green[Search Result]")
            st.write(result['result'])
    
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
    st.session_state.history.append(query)

history = st.sidebar.expander("Query History")
for i, past_query in enumerate(st.session_state.history, start = 1): 
    history.write(f"**{i}** - :red[{past_query}]")