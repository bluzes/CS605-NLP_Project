import streamlit as st
import requests

from llama_index import VectorStoreIndex, SimpleDirectoryReader, TreeIndex, ListIndex
from llama_index import StorageContext, load_index_from_storage, GPTVectorStoreIndex
from llama_index import LLMPredictor, PromptHelper, ServiceContext, Document, ResponseSynthesizer
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.indices.struct_store import PandasIndex, GPTPandasIndex
from llama_index.indices.document_summary import DocumentSummaryIndex
from langchain.chat_models import ChatOpenAI
from llama_index.indices.document_summary import DocumentSummaryIndexRetriever
from llama_index import MockLLMPredictor, ServiceContext, MockEmbedding
import openai

import os
import pandas as pd

from langchain import OpenAI
import logging
import sys

openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Flask API endpoint
# API_URL = 'http://localhost:5000' # No flask required atm

# Header
st.header('Finasse')

# Title
st.title('Welcome to Finasse! Your personal Financial Assistant for analysis')

# Placeholder body content
st.write('This is the body of my app.')

# File uploader
file = st.file_uploader('Upload a file')

if file is not None:
    pass
    # Process the uploaded file
    # Your code here

# Two separated text box inputs
input1 = st.text_input('Enter text input 1')
input2 = st.text_input('Enter text input 2')
# submit1 = st.button('Submit1')
# submit2 = st.button('Submit2')


# # Submit button
# if st.button('Submit1'):
#     # Send the inputs to the Flask API
#     data = {'input1': input1, 'input2': input2}
#     response = requests.post(f'{API_URL}/process', json=data)
    
#     # Handle the response from the Flask API
#     if response.status_code == 200:
#         result = response.json()
#         st.write('API Response:', result)
#     else:
#         st.error('Error occurred during API request.')

# if st.button('Submit2'):
#     # Send the inputs to the Flask API
#     data = {'input1': input1, 'input2': input2}
#     response = requests.post(f'{API_URL}/process', json=data)
    
#     # Handle the response from the Flask API
#     if response.status_code == 200:
#         result = response.json()
#         st.write('API Response:', result)
#     else:
#         st.error('Error occurred during API request.')

from langchain.chat_models import ChatOpenAI

index_name = "./saved_index/new"
preloaded_index = "./saved_index/preloaded"
pre_loaded_documents_folder = "./datasets/preloaded/small"
documents_folder = "./datasets/csv"

print("Testing Printing to Streamlit Logs")

def preload_index():
    documents = SimpleDirectoryReader(pre_loaded_documents_folder).load_data()
    llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=2048)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=preloaded_index)
        index = load_index_from_storage(storage_context,index_id="Pre_Loaded_Small_2_Docs") # this index should contain the full documents from through documents 1 & 2
        # doc_summary_index = load_index_from_storage(storage_context, index_id ="3a995849-05eb-433a-8b81-7155b52c33c5") # this index should contain the summary
        print("Loaded")
        print("Checking if index is properly loaded")
        if index:
            print(index)
        else:
            print("Index Not Loaded")
        print("----- End of Check -------")
        st.write("Index and Doc Summary Index Loaded")
        return index
    except Exception as e:
        # index = GPTVectorStoreIndex([])
        # index.insert(documents[0])
        print("Exception: ",e)
        st.write(e)
        return index


@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_name)

    return index



# @st.cache_data(max_entries=200, persist=True)
# def query_index(_index, query_text):
#     if _index is None:
#         return "Please initialize the index!"

#     # query_engine = index.as_query_engine(response_mode= "compact",verbose=True,similarity_top_k=5)
#     # response = query_engine.query("Keypoints of performance comparison of all given companies in Q4 2021")
#     # print(response)
#     response = _index.as_query_engine(response_mode = "compact",verbose = True, similarity_top_k = 3).query(query_text)
#     print(response)
#     st.write(response)
#     return str(response)

preloaded_doc_summary_index = None
preloaded_index = preload_index()
# index = initialize_index(index_name, documents_folder)


# submit3 = st.button('Run Query')

col1, col2 = st.columns(2)

temp_min_value = 0.0
temp_max_value = 1.0
temp_default_value = 0.0
step = 0.1
temp_value = col2.slider('Select a value', temp_min_value, temp_max_value, temp_default_value, step = step)
col2.text('Temperature value:', temp_value)


k_min_value = 2
k_max_value = 10
k_default_value = 2
k_value = col2.slider('Select a value', k_min_value, k_max_value, k_default_value)
col2.text('Top-K value:', k_value)



# Define the options for the dropdown
options = {'default':'refine','compact':'compact', 'tree summary':'tree_summarize', 'mock run':'no_text'}

# Create the dropdown box
selected_option = col1.selectbox('Query mode option', options.keys())
selected_value = options[selected_option]

# Display the selected option
col1.text('Selected option:', selected_option)


# Testing debugging
# st.write("Debugging Starts")
# st.write(type(preload_index))
# st.write(preload_index)
# # st.write(preload_index.docstore.docs)
# st.write("*** End of Debug")
# End Test
text = st.text_input("Query text:", value="Keypoints of Disney Q4 performance")




if st.button("Run Query") and text is not None:
    print("Checking if index is working")
    query_engine = preloaded_index.as_query_engine(response_mode= selected_value,verbose=True,similarity_top_k=k_value)
    response = query_engine.query(text)
    st.markdown(f'<div style="width: 40em; height: 200px; word-wrap: break-word; overflow-y: auto;">{response}</div>',unsafe_allow_html=True)
    #     f'<div style="width: 300px; height: 200px; word-wrap: break-word; overflow-y: auto;">{text}</div>',
    # <div style="width: 40em;height: auto;word-wrap: break-word;white-space: break-spaces;overflow: scroll;">
    # Disney's fourth quarter performance was strong, with operating income increasing by $1.6 billion year-over-year. All of Disney's sites were open for the entire quarter, although generally at reduced capacities. Attendance trends continued to strengthen at Disney's domestic parks, with Walt Disney World Q4 attendance up double-digits versus Q3, and Disneyland attendance continuing to strengthen significantly from its reopening in the third quarter. Guest spending at Disney's domestic parks was up nearly 30% versus fiscal 2019. Disney Cruise Line returned to sea with guest ratings as strong as pre-pandemic levels. Disney's Media and Entertainment Distribution segment saw a decrease in operating income of approximately $600 million versus the prior year, driven by lower results at Linear Networks, Direct-to-Consumer and content sales, licensing, and other. Consumer Products operating results declined in the fourth quarter, impacted by a tough comparison in Disney's games business.</div>
    
    # {
    # width: 40em;
    # height: auto;
    # word-wrap: break-word;
    # white-space: break-spaces;
    # overflow: scroll;
    # }



    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Tokens Used: {preloaded_index.service_context.llm_predictor._last_token_usage}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {preloaded_index.service_context.embed_model._last_token_usage}"
        )
# else:
#     print("Checking stuff")
#     print(text)
#     print("******END*******")

# if 'response' not in st.session_state:
#     st.session_state.response = ''


# def run_query():
#     st.session_state.response = query_engine.query(text)

# st.button("Run Query", on_click=run_query)
# if st.session_state.response:
#     st.subheader("Response: ")
#     st.success(st.session_state.response)

