from query_vector_db import rag_chain
import streamlit as st

st.write(
    '''
     # Query the PDF using Hybrid Search and LLM
    '''
)

query = st.text_input('Enter the query...')
    
if (st.button('Result')):
    result = rag_chain.invoke(query)
    st.write(result)