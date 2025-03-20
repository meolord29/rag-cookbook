# Import Streamlit for UI creation
import streamlit as st

# Set the Streamlit page title and icon; this must be done before other Streamlit commands.
st.set_page_config(page_title="RAG Chat Interface", page_icon="🤖")  

# Import required libraries/modules for processing
import ast
import pandas as pd
from openai import OpenAI  # OpenAI library for interacting with language models
from qdrant_client import QdrantClient, models  # Used for vector database operations
import os
import torch

# Temporary workaround for a potential module path conflict in torch
torch.classes.__path__ = []

# Function to lazily load the SentenceTransformer embedding model and cache the resource for optimization
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer  # Deferred import to optimize resource usage
    return SentenceTransformer('BAAI/bge-large-zh-v1.5')  # Using a pre-trained embedding model specific to Chinese language

# Function to initialize a Qdrant vector database in-memory with pre-loaded data; cached for efficiency
@st.cache_resource
def init_qdrant():
    """Initialize Qdrant client with pre-loaded data"""
    embedding_input = "/app/data/dataset_ETL3.csv"  # Path to embedding dataset
    
    # Load CSV file containing embeddings and accompanying metadata
    df = pd.read_csv(embedding_input)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)  # Convert string representation of embedding to Python list
    
    # Create an in-memory Qdrant client to host the vector database
    client = QdrantClient(":memory:")
    vector_size = len(df['embedding'].iloc[0])  # Get size of the embedding vectors from the dataset
    
    # Define the vector database collection with specified configurations
    client.create_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(
            size=vector_size,  # Number of dimensions in embeddings
            distance=models.Distance.COSINE  # Cosine similarity is used for vector comparison
        )
    )
    
    # Upload dataset points into the Qdrant vector database, including embedding vectors and metadata
    client.upload_points(
        collection_name="documents",
        points=[
            models.PointStruct(
                id=idx,  # Unique identifier for each point
                vector=row['embedding'],  # Embedding vector
                payload={
                    "text": row['chunk'],  # Text associated with the embedding
                    "source": row['relative_path']  # Source path for the text
                }
            )
            for idx, row in df.iterrows()  # Iterate through each row in the dataset
        ]
    )
    
    return client  # Return the initialized Qdrant client

# Initialize main application components (Qdrant and embedding model)
qdrant_client = init_qdrant()
embedding_model = load_embedding_model()

# Set up the main title of the application interface
st.title("Custom LLM RAG Assistant")

# Initialize chat history in the session state; the "system" message contains initial instructions for the assistant
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": "You're a helpful assistant. Use the provided context to answer questions accurately. Before giving any response tell the user where you have found the source of interest, then give an answer based on the reference source"
    }]

# Sidebar UI configuration for setting API parameters and resetting chat history
with st.sidebar:
    st.header("API Configuration")  # Sidebar title
    
    # Input fields for API base URL and API key
    api_base = st.text_input(
        "API Base URL:",
        value="http://dev_litellm.myhomelab.org",  # Default value for API URL
        placeholder="http://dev_litellm.myhomelab.org"
    )
    api_key = st.text_input("API Key:", type="password")  # Password input for API key
    
    # Dropdown menu to select language model and slider to select search result limit
    selected_model = st.selectbox("Choose Model", ("fake-1", "gpt-4o"), index=0)  # Predefined models
    search_limit = st.slider("Context Chunks to Use", 1, 10, 5)  # Slider to select chunk limit for vector search
    
    # Button to reset the chat history in session
    if st.button("Reset Chat"):
        st.session_state.messages = []

# Display chat