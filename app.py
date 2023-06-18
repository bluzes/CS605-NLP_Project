import streamlit as st
import requests

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
