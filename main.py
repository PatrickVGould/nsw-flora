"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import Pinecone
import pinecone

st.set_page_config(page_title="NSW Flora Finder", page_icon=":robot:")
st.header("NSW Flora Finder")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]


def qa_source_vector(model = 'OpenAI', index_name='nsw-plants'):
    if model == 'ChatOpenAI':
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    else:
        llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)

    pinecone.init(api_key=pinecone_api_key,environment='us-east-1-aws')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    new_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='map_reduce', vectorstore=docsearch)
    return new_chain

def get_text():
    input_text = st.text_input("You: ", "Input your text description of a NSW Flora Species?", key="input")
    return input_text

new_chain = qa_source_vector()

user_input = get_text()

if user_input:
    output = new_chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
