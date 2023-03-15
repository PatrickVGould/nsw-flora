"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone



openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = st.secrets["PINECONE_INDEX_NAME"]

def qa_source_vector():
    llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    new_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='map_reduce', vectorstore=docsearch)
    return new_chain

new_chain = qa_source_vector()

#From here all down is streamlit

st.set_page_config(page_title="NSW Flora Finder 🌿🍃🌱", page_icon="favicon.ico")
st.header("NSW Flora Finder 🌿🍃🌱")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input", placeholder="Input a question about NSW Flora: 🍃")
    return input_text

user_input = get_text()

if user_input:
    try:
        output = new_chain.run(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    except:
        st.session_state.past.append(user_input)
        st.session_state.generated.append("Question returned with error. Try changing the question or make it more specific.")

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i),
                avatar_style = "bottts-neutral",
                seed='qweqdsa'
                )
        message(st.session_state["past"][i], is_user=True, avatar_style = 'fun-emoji', seed = 'asdascd', key=str(i) + "_user")

