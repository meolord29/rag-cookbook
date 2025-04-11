# Simple RAG Implementation

This recipe demonstrates how to build a basic Retrieval-Augmented Generation (RAG) system that enhances LLM responses with relevant information from your own documents.

## Questions This Recipe Answers
- How do I put my data into my app?
- How do I ingest data into my app?
- How do I "embed" documents?
- How do I set up a vector database?
- How do I incorporate a vector database into the application?

## Key Components

### 1. Document Processing
The recipe shows how to ingest, parse, and prepare documents for embedding and retrieval.

**Implementation:**
```python
def process_documents(file_paths):
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    return documents
```

### 2. Text Chunking
Implements document chunking to break down large texts into manageable pieces for embedding and retrieval.

**Benefits of Chunking:**
- Improves retrieval precision
- Ensures context fits within token limits
- Enables more focused responses

**Implementation:**
```python
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### 3. Document Embedding
Converts text chunks into vector embeddings for semantic search.

**Implementation:**
```python
def embed_documents(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectors = []
    for chunk in tqdm(chunks, desc="Embedding documents"):
        embedding = embeddings.embed_query(chunk.page_content)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown")
            }
        })
    return vectors
```

### 4. Vector Database Integration
Stores and retrieves document embeddings using a vector database (Qdrant).

**Features:**
- In-memory or persistent storage options
- Semantic similarity search
- Metadata filtering capabilities
- Scalable architecture

### 5. RAG Query Pipeline
Combines vector search with LLM generation to create context-aware responses.

**Implementation:**
```python
def query_rag_system(query, collection, llm):
    # Get embeddings for the query
    query_embedding = embeddings.embed_query(query)
    
    # Search the vector database
    search_results = collection.search(
        query_vector=query_embedding,
        limit=5
    )
    
    # Extract retrieved documents
    retrieved_docs = [hit.metadata["text"] for hit in search_results]
    context = "\n\n".join(retrieved_docs)
    
    # Generate response with context
    messages = [
        {"role": "system", "content": f"Use the following context to answer the user's question: {context}"},
        {"role": "user", "content": query}
    ]
    
    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content
```

## Implementation Details

This recipe demonstrates the essential components of a RAG system:

1. **Document Ingestion**: Loading and processing documents from various sources
2. **Text Chunking**: Breaking documents into manageable pieces
3. **Embedding Generation**: Converting text to vector representations
4. **Vector Storage**: Setting up and using a vector database
5. **Retrieval Pipeline**: Finding relevant documents based on user queries
6. **Response Generation**: Using retrieved context to enhance LLM responses

## Usage

To run the simple RAG application:

1. Install the required dependencies:
   ```bash
   pip install qdrant-client sentence-transformers tqdm pandas docling langchain-text-splitters streamlit openai
   ```

2. Process and embed your documents using the provided scripts

3. Run the Streamlit app to interact with your RAG system:
   ```bash
   streamlit run app/simple_rag_app.py
   ```

## References
1. [Docling Documentation](https://github.com/docling-project/docling/tree/main/docs) - Information about Docling, how to parse documents
2. [BGE Embedding Model](https://huggingface.co/BAAI/bge-large-en-v1.5) - Straightforward small embedding model, fairly good as well
3. [Sentence Transformers](https://sbert.net/) - The Python package for quick embedding functionality, super simple to use
4. [Qdrant Vector Database](https://qdrant.tech/documentation/) - Great vector database that can not only run in memory but is feature-rich, and can easily run on a server or cloud without changing any of the code
5. [Document Splitting Guide](https://dev.to/rutamstwt/langchain-document-splitting-21im) - Recursive text splitting with respect to markdown annotations