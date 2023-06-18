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

import os
import pandas as pd

from langchain import OpenAI
import logging
import sys

# Flask API endpoint
API_URL = 'http://localhost:5000'

# Header
st.header('My Streamlit App')

# Title
st.title('Welcome to my app!')

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

# Submit button
if st.button('Submit'):
    # Send the inputs to the Flask API
    data = {'input1': input1, 'input2': input2}
    response = requests.post(f'{API_URL}/process', json=data)
    
    # Handle the response from the Flask API
    if response.status_code == 200:
        result = response.json()
        st.write('API Response:', result)
    else:
        st.error('Error occurred during API request.')
