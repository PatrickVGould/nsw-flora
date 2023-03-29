"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import pyalex


pyalex.config.email = "shoerac97@gmail.com"
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = st.secrets["PINECONE_INDEX_NAME"]

def paper_search(query):
    # set your email to get into the polite pool
    
    # set the number of results to retrieve per page
    per_page = 10
    
    # search for articles related to a query
    search_results = pyalex.Works().search(query).paginate(per_page=per_page)
    print(search_results)
    
    # get all results by iterating over pages until there are no more pages
    results = []
    pages_retreived = 0
    for page in search_results:
        if pages_retreived == 1:
            break
        results += page
        pages_retreived += 1
    num_results = len(results)
    print(results)
    #print(f"Found {num_results} results")
    papers = []
    i=0
    for paper in results:
        print(paper.get('abstract', ''))
        title = paper.get('title', '')
        abstract = paper.get('abstract_inverted_index', '')
        if abstract != '':
            abstract = pyalex.invert_abstract(abstract)
        date = paper.get("publication_date", "")[:4]
        papers.append({
            "title": title,
            "abstract": abstract,
            "date": date
        })

    print(papers)
    return papers

def qa_source_vector(query):
    llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    new_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='map_reduce', vectorstore=docsearch)
    return new_chain({'question':query}, return_only_outputs=False)

tools = [
    Tool(
        name = "Academic Search",
        func=paper_search,
        description="useful for when you need to find a research paper to answer a question about research. The input to this should be a single, simple boolean search term that will be put into google scholar."
    ),
    Tool(
        name = "Query Vector Database",
        func=qa_source_vector,
        description="useful for when you need to query a vector database that contains information on Australian plants. The input to this should be a single, simple query that would provide a high similarity score to the information you are after."
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm_agent=ChatOpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm_agent, agent="chat-conversational-react-description", verbose=False, memory=memory)

def run_agent(user_input):
    response = agent_chain.run(input=user_input)
    return response

#From here all down is streamlit

st.set_page_config(page_title="FloraAustralisChat 🔍", page_icon="favicon.ico")
st.header("FloraAustralisChat 🔍🌿🍃🌱")
with st.sidebar:
    st.caption("""Welcome to the FloraAustralisChat! 🔍🌿🍃🌱  \n   \n This tool helps you find information about the flora in New South Wales.   \n   \n To get started, simply type your question about NSW flora in the input field and press Enter. Keep in mind that you are interacting with an AI language model, which can provide accurate and relevant information most of the time but may occasionally produce unexpected or incorrect responses due to the complexities of natural language understanding. If you don't receive a satisfactory answer, try rephrasing your question or making it more specific.  \n   \n The AI model is trained on a range of publically available Australian Flora documents. It has also been trained on the wikipedia pages for the category 'Flora of New South Wales'  \n (https://en.wikipedia.org/wiki/Category:Flora_of_New_South_Wales)  \n  It can answer questions about the flora in NSW and Australia, but it may not be able to answer questions about other topics.  \n Note that it the training documents are inconsistant and so it may have much more information on some plants than others. As such, answers on some topics that will be limited or non-existant.   \n   \n If you have any questions or feedback, please contact me at patrickvgould97@gmail.com""")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input", placeholder="Ask a question relating to Australian Flora: 🔍🍃")
    return input_text

user_input = get_text()

if user_input:
    try:
        output = run_agent(user_input)
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

