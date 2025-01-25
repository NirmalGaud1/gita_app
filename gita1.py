#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure models
genai.configure(api_key="AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c")
gemini = genai.GenerativeModel('gemini-pro')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('bhagwatgeeta.csv')
    
    # Combine relevant text for embedding
    df['context'] = df.apply(lambda row: f"Chapter {row['Chapter']} Verse {row['Verse']}: {row['EngMeaning']}", axis=1)
    
    # Generate embeddings
    embeddings = embedder.encode(df['context'].tolist(), show_progress_bar=False)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    return df, index

df, faiss_index = load_data()

# Simple UI
st.title("Bhagavad Gita RAG Chatbot")
st.subheader("Ask any question related to Gita teachings")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# RAG Pipeline
def get_gita_answer(query):
    # Step 1: Retrieve relevant verses
    query_embedding = embedder.encode([query])
    D, I = faiss_index.search(query_embedding.astype('float32'), k=3)
    
    # Get top passages
    contexts = [df.iloc[i]['context'] for i in I[0]]
    
    # Step 2: Generate answer with Gemini
    prompt = f"""
    You are a Bhagavad Gita expert. Answer the question using ONLY the context below.
    Keep response under 100 words. Cite chapter/verse numbers when possible.
    
    Context:
    {''.join([f'- {c}\n' for c in contexts])}
    
    Question: {query}
    Answer:
    """
    
    response = gemini.generate_content(prompt)
    return response.text

# User input
if prompt := st.chat_input("Ask your question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Consulting the Gita..."):
        response = get_gita_answer(prompt)
    
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update messages
    st.rerun()

