# Advanced RAG System with Re-ranking and Relevancy Evaluation
# This application implements a sophisticated RAG system with multiple stages of retrieval refinement

# ================ IMPORTS ================
# Core libraries
import streamlit as st
import pandas as pd
import ast
import os
import re
import time
import torch

# Vector database
from qdrant_client import QdrantClient, models

# Language models and embeddings
from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.evaluation import RelevancyEvaluator
from sentence_transformers import CrossEncoder

# ================ CONFIGURATION ================
# Set the Streamlit page configuration
st.set_page_config(
    page_title="Advanced RAG Assistant", 
    page_icon="🤖",
    layout="wide"
)

# Fix for torch module path conflict
torch.classes.__path__ = []

# ================ MODEL LOADING ================
# All model loading functions are cached for efficiency

@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model for vector similarity search"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('BAAI/bge-large-zh-v1.5')

@st.cache_resource
def load_reranker_model():
    """Load and cache the cross-encoder model for re-ranking"""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def init_qdrant():
    """Initialize and populate in-memory Qdrant vector database with pre-embedded data"""
    # Configuration
    embedding_input = "./app/data/dataset_ETL3.csv"
    collection_name = "documents"
    
    # Load embeddings from CSV
    df = pd.read_csv(embedding_input)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    
    # Initialize in-memory vector database
    client = QdrantClient(":memory:")
    vector_size = len(df['embedding'].iloc[0])
    
    # Create vector collection with cosine similarity
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    
    # Upload dataset points into the vector database
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=row['embedding'],
                payload={
                    "text": row['chunk'],
                    "source": row['relative_path']
                }
            )
            for idx, row in df.iterrows()
        ]
    )
    
    return client  # Return the initialized Qdrant client

# ================ RETRIEVAL AND RANKING FUNCTIONS ================

def retrieve_documents(query, client, embedding_model, top_k=5):
    """Retrieve documents from vector database based on query embedding"""
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query).tolist()
    
    # Search for similar documents in the vector database
    search_results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Extract and format the search results
    documents = []
    for result in search_results:
        documents.append({
            "text": result.payload["text"],
            "source": result.payload["source"],
            "score": float(result.score)
        })
    
    return documents

def rerank_documents(query, documents, reranker_model):
    """Re-rank documents using a cross-encoder model for more accurate relevance assessment"""
    # Prepare query-document pairs for scoring
    pairs = [[query, doc["text"]] for doc in documents]
    
    # Get relevance scores from cross-encoder
    scores = reranker_model.predict(pairs)
    
    # Create new document list with updated scores
    scored_docs = []
    for i, doc in enumerate(documents):
        scored_docs.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": float(scores[i])
        })
    
    # Sort by score in descending order
    return sorted(scored_docs, key=lambda x: x["score"], reverse=True)

def evaluate_relevance(query, documents, llm):
    """Evaluate document relevance using direct LLM scoring"""
    evaluated_docs = []
    
    for doc in documents:
        try:
            # Create prompt for relevance scoring
            prompt = f"""Rate the relevance of the following text to the query on a scale of 0 to 10, 
            where 0 means completely irrelevant and 10 means perfectly relevant.
            
            Query: {query}
            
            Text: {doc['text']}
            
            Relevance score (just the number between 0-10):"""
            
            # Get LLM's relevance assessment
            response = llm.complete(prompt)
            
            # Extract numeric score using regex
            score_match = re.search(r'\b([0-9]|10)\b', response.text)
            
            if score_match:
                # Normalize score to 0-1 range
                raw_score = float(score_match.group(1))
                relevance_score = raw_score / 10.0
            else:
                relevance_score = 0.5  # Default if no score found
                
            # Display debug information in sidebar
            st.sidebar.write(f"Document: {doc['source'][:30]}...")
            st.sidebar.write(f"Query: {query}")
            st.sidebar.write(f"LLM response: {response.text}")
            st.sidebar.write(f"Relevance score: {relevance_score:.2f}")
            st.sidebar.write("---")
            
            # Prevent rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            st.warning(f"Relevancy evaluation error: {str(e)}")
            relevance_score = doc["score"]  # Fallback to original score
        
        # Add document with relevance score
        evaluated_docs.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": doc["score"],
            "relevance": relevance_score
        })
    
    return evaluated_docs

# ================ APPLICATION INITIALIZATION ================
# Initialize all required components
qdrant_client = init_qdrant()
embedding_model = load_embedding_model()
reranker_model = load_reranker_model()

# ================ UI SETUP ================
# Main application title
st.title("Advanced RAG Assistant")
st.markdown("#### With Re-ranking and Relevancy Evaluation")

# Initialize chat history with system instructions
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": "You're a helpful assistant. Use the provided context to answer questions accurately. "
                  "Begin your response by citing the source documents you're using, then provide a comprehensive answer."
    }]

# ================ SIDEBAR CONFIGURATION ================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API settings section
    st.subheader("API Settings")
    api_base = st.text_input(
        "API Base URL",
        value="http://dev_litellm.myhomelab.org",
        placeholder="Enter API endpoint URL"
    )
    api_key = st.text_input("API Key", type="password", help="Your API key for authentication")
    
    # Model selection
    st.subheader("Model Settings")
    selected_model = st.selectbox(
        "Language Model", 
        options=["fake-1", "gpt-4o"], 
        index=0,
        help="Select the language model to use for responses"
    )
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    search_limit = st.slider(
        "Initial Retrieval Count", 
        min_value=5, 
        max_value=30, 
        value=20,
        help="Number of documents to retrieve in the initial search"
    )
    
    # Re-ranking settings
    use_reranking = st.checkbox(
        "Enable Re-ranking", 
        value=True,
        help="Use cross-encoder model to re-rank retrieved documents"
    )
    
    if use_reranking:
        reranked_limit = st.slider(
            "Documents After Re-ranking", 
            min_value=3, 
            max_value=10, 
            value=5,
            help="Number of re-ranked documents to use for the final response"
        )
    
    # Relevancy evaluation settings
    use_relevancy_eval = st.checkbox(
        "Enable Relevancy Evaluation", 
        value=True,
        help="Use LLM to evaluate document relevance to the query"
    )
    
    # Add a divider for visual separation
    st.divider()
    
    # Button to reset the chat history in session
    if st.button("Reset Chat"):
        st.session_state.messages = []

# ================ CHAT INTERFACE ================
# Display chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ================ QUERY PROCESSING ================
# Process user input when submitted
if prompt := st.chat_input("Type your question..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        # Validate API configuration
        if not api_base or not api_key:
            st.error("⚠️ Please configure API settings in the sidebar!")
            st.stop()
        
        # Create placeholders for response and status updates
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Initialize API clients
            client = OpenAI(base_url=api_base, api_key=api_key)
            llama_index_llm = LlamaIndexOpenAI(model=selected_model, api_key=api_key, api_base=api_base)
            
            # ================ DOCUMENT RETRIEVAL ================
            status_placeholder.markdown("🔍 Retrieving relevant documents...")
            
            # Get initial documents using vector similarity
            retrieved_docs = retrieve_documents(
                query=prompt, 
                client=qdrant_client, 
                embedding_model=embedding_model, 
                top_k=search_limit
            )
            
            # Store original ranking for comparison
            original_docs = retrieved_docs.copy()
            
            # ================ DOCUMENT RE-RANKING ================
            if use_reranking:
                status_placeholder.markdown("🔄 Re-ranking documents for better relevance...")
                retrieved_docs = rerank_documents(prompt, retrieved_docs, reranker_model)
                retrieved_docs = retrieved_docs[:reranked_limit]  # Limit to top results
                
                # Display re-ranking visualization
                with st.expander("📊 Re-ranking Analysis"):
                    st.markdown("### Document Ranking Transformation")
                    st.markdown("""
                    **Initial Retrieval**: Vector similarity search using sentence embeddings
                    **Re-ranking**: Direct query-document comparison using cross-encoder model
                    """)
                    
                    # Build comparison table
                    comparison_data = []
                    top_original = original_docs[:reranked_limit]
                    
                    for i, reranked_doc in enumerate(retrieved_docs):
                        # Find original position
                        original_pos = next((j for j, d in enumerate(original_docs) if d["text"] == reranked_doc["text"]), -1)
                        position_change = original_pos - i if original_pos >= 0 else "N/A"
                        
                        # Get document from original ranking at same position for comparison
                        orig_doc = top_original[i] if i < len(top_original) else {"source": "N/A", "score": 0, "text": ""}
                        
                        # Prepare text previews
                        orig_text = orig_doc["text"][:100] + "..." if len(orig_doc["text"]) > 100 else orig_doc["text"]
                        reranked_text = reranked_doc["text"][:100] + "..." if len(reranked_doc["text"]) > 100 else reranked_doc["text"]
                        
                        # Add to comparison data
                        comparison_data.append({
                            "Rank": i+1,
                            "Original Source": orig_doc["source"],
                            "Original Score": f"{orig_doc['score']:.4f}",
                            "Original Text": orig_text,
                            "Re-ranked Source": reranked_doc["source"],
                            "Re-ranked Score": f"{reranked_doc['score']:.4f}",
                            "Re-ranked Text": reranked_text,
                            "Position Change": f"{position_change}" if position_change != "N/A" else "N/A"
                        })
                    
                    # Display as dataframe
                    st.dataframe(pd.DataFrame(comparison_data))
            
            # ================ RELEVANCY EVALUATION ================
            if use_relevancy_eval:
                # Store pre-evaluation ranking
                pre_eval_docs = retrieved_docs.copy()
                
                status_placeholder.markdown("⚖️ Evaluating document relevance using LLM...")
                retrieved_docs = evaluate_relevance(prompt, retrieved_docs, llama_index_llm)
                
                # Sort by relevance score and limit results
                retrieved_docs.sort(key=lambda x: x["relevance"], reverse=True)
                retrieved_docs = retrieved_docs[:reranked_limit]
                
                # Display relevancy evaluation visualization
                with st.expander("🔍 Relevancy Evaluation Analysis"):
                    st.markdown("### LLM-Based Relevance Assessment")
                    st.markdown("""
                    Documents are evaluated by an LLM for direct relevance to your query.
                    This provides a more nuanced assessment beyond statistical similarity.
                    """)
                    
                    # Build comparison table
                    eval_comparison = []
                    
                    for i, doc in enumerate(retrieved_docs):
                        # Find position before evaluation
                        original_pos = next((j for j, d in enumerate(pre_eval_docs) if d["text"] == doc["text"]), -1)
                        position_change = original_pos - i if original_pos >= 0 else "N/A"
                        
                        # Text preview
                        doc_text = doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
                        
                        # Add to comparison data
                        eval_comparison.append({
                            "New Rank": i+1,
                            "Previous Rank": original_pos+1 if original_pos >= 0 else "N/A",
                            "Source": doc["source"],
                            "Original Score": f"{doc['score']:.4f}",
                            "Relevance Score": f"{doc['relevance']:.4f}",
                            "Text Preview": doc_text,
                            "Position Change": f"{position_change}" if position_change != "N/A" else "N/A"
                        })
                    
                    # Display as dataframe
                    st.dataframe(pd.DataFrame(eval_comparison))
            
            # ================ RESPONSE GENERATION ================
            # Format context from retrieved documents with document IDs for reference
            context_with_ids = []
            for i, doc in enumerate(retrieved_docs):
                doc_id = f"[DOC-{i+1}]"
                doc["doc_id"] = doc_id  # Add ID to document for later reference
                context_with_ids.append(f"{doc_id} Source: {doc['source']}\n\n{doc['text']}")
            
            context = "\n\n---\n\n".join(context_with_ids)
            
            # Prepare messages for chat completion
            messages = [
                {"role": "system", "content": """You're a helpful assistant. Use the provided context to answer questions accurately.
                
                IMPORTANT INSTRUCTIONS FOR CITATIONS:
                1. Each document in the context is marked with a document ID like [DOC-1], [DOC-2], etc.
                2. Begin your response with a brief summary of which documents you're using.
                3. Throughout your response, cite specific documents using their document IDs in brackets.
                4. At the end of your response, include a 'References' section that lists all documents you cited.
                5. For each citation, explain which part of your answer used information from that document.
                
                Example format for references section:
                
                REFERENCES:
                - [DOC-1]: Used for information about X in paragraphs 1-2
                - [DOC-3]: Used for details about Y in paragraph 3
                """}
            ]
            messages.append({"role": "user", "content": f"Context information:\n\n{context}\n\nQuestion: {prompt}"})

            
            # Generate streaming response
            status_placeholder.markdown("💬 Generating response...")
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True
            )
            
            # Display streaming response
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            # Display final response and update chat history
            status_placeholder.empty()
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"🚨 API Error: {str(e)}")
            if "geographic" in str(e).lower():
                st.info("🌐 Try using a VPN connection to supported regions")
            st.stop()
