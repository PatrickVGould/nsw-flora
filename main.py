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
with st.sidebar:
    st.caption("Welcome to the NSW Flora Finder! 🌿🍃🌱 This tool helps you find information about the flora in New South Wales. To get started, simply type your question about NSW flora in the input field and press Enter. Keep in mind that you are interacting with an AI language model, which can provide accurate and relevant information most of the time but may occasionally produce unexpected or incorrect responses due to the complexities of natural language understanding. If you don't receive a satisfactory answer, try rephrasing your question or making it more specific. The AI model is trained on most of the volumes of 'The Flora of Australia' series which includes all flowering and non-flowering plants known to be indigenous or naturalised in Australia. It has also been trained on the wikipedia pages for the categora 'Flora of NSW'. It can answer questions about the flora in NSW and Australia, but it may not be able to answer questions about other topics. Note that it has not been trained on the following Flora of Australia volumes as they are not available online: Volume 8 - Lecythidales to Batales, Volume 39 - Alismatales to Arale, Volume 43 - Poaceae 1, Volume 44A - Poaceae 2, Volume 44B - Poaceae 3, Volume 54 - Lichens - Introduction, Lecanorales 1, Volume 55 - Lichens - Lecanorales 2, Parmeliaceae, Volume 56A - Lichens 4, Volume 57 - Lichens 5, and Volume 58A - Lichens 3. As such, answers on topics that would be contained in these volumes will be limited or non-existant. If you have any questions or feedback, please contact me at patrickvgould97@gmail.com")


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
        st.session_state.generated.append("Question returned with error (It was likely too complex and required me to read too much information). Try changing the question or make it more specific.")

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i),
                avatar_style = "bottts-neutral",
                seed='qweqdsa'
                )
        message(st.session_state["past"][i], is_user=True, avatar_style = 'fun-emoji', seed = 'asdascd', key=str(i) + "_user")

