# Advanced RAG System with Re-ranking, Relevancy Evaluation, and Source Attribution

This recipe demonstrates how to implement a sophisticated RAG (Retrieval-Augmented Generation) system that improves retrieval accuracy through re-ranking, relevancy evaluation, and transparent source attribution.

## Questions This Recipe Answers
- How do I make my retrieval more accurate?
- What is a re-ranker and how do I implement it?
- When should I use a re-ranker?
- How can I implement a relevancy evaluator agent?
- How can I track which sources are used in which parts of the generated response?
- How do I visualize ranking changes for better explainability?

## Key Components

### 1. Re-ranking
Re-ranking is a technique that improves retrieval quality by applying a more sophisticated model to re-score the initial set of retrieved documents. While the initial vector search uses embedding similarity, re-ranking can consider the semantic relationship between the query and document more deeply.

**Benefits of Re-ranking:**
- Improves precision of retrieved documents
- Reduces hallucinations by ensuring more relevant context
- Allows for retrieving a larger initial set of documents and then filtering to the most relevant ones

**Implementation:**
```python
def rerank_documents(query, documents, reranker_model):
    # Create query-document pairs for the cross-encoder
    pairs = [[query, doc["text"]] for doc in documents]
    # Score each pair using the cross-encoder model
    scores = reranker_model.predict(pairs)
    # Combine scores with document metadata
    scored_docs = []
    for i, doc in enumerate(documents):
        scored_docs.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": float(scores[i])
        })
    # Sort by score in descending order
    return sorted(scored_docs, key=lambda x: x["score"], reverse=True)
```

### 2. Relevancy Evaluation
The relevancy evaluator agent uses LLM-based evaluation to determine how relevant each document is to the user's query. This provides an additional layer of filtering beyond vector similarity and re-ranking scores.

**Benefits of Relevancy Evaluation:**
- Provides a more nuanced assessment of document relevance
- Can filter out documents that are semantically similar but not actually relevant
- Helps prioritize the most informative documents for the LLM context

**Implementation:**
```python
def evaluate_relevance(query, documents, llm):
    evaluated_docs = []
    for doc in documents:
        # Create a prompt asking the LLM to rate relevance on a scale of 0-10
        prompt = f"""Rate the relevance of the following text to the query on a scale of 0 to 10, 
        where 0 means completely irrelevant and 10 means perfectly relevant.
        Query: {query}
        Text: {doc['text']}
        Relevance score (just the number between 0-10):"""
        # Get the LLM's response
        response = llm.complete(prompt)
        # Extract the numerical score
        score_match = re.search(r'\b([0-9]|10)\b', response.text)
        if score_match:
            raw_score = float(score_match.group(1))
            relevance_score = raw_score / 10.0
        else:
            relevance_score = 0.5  # Default if no score found
        # Add relevance score to document metadata
        evaluated_docs.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": doc["score"],
            "relevance": relevance_score
        })
    return evaluated_docs
```

### 3. Transparent Source Attribution
The system tracks which sources are used in the generated response by assigning unique identifiers to each document and instructing the LLM to cite these sources throughout its response.

**Benefits of Source Attribution:**
- Increases transparency and trustworthiness of the generated response
- Allows users to verify information against source documents
- Provides clear traceability between response content and source material

**Implementation:**
```python
# Format context from retrieved documents with document IDs for reference
context_with_ids = []
for i, doc in enumerate(retrieved_docs):
    doc_id = f"[DOC-{i+1}]"
    doc["doc_id"] = doc_id  # Add ID to document for later reference
    context_with_ids.append(f"{doc_id} Source: {doc['source']}\n\n{doc['text']}")

context = "\n\n---\n\n".join(context_with_ids)

# Instruct the LLM to use citations
messages = [
    {"role": "system", "content": """You're a helpful assistant. Use the provided context to answer questions accurately.
    
    IMPORTANT INSTRUCTIONS FOR CITATIONS:
    1. Each document in the context is marked with a document ID like [DOC-1], [DOC-2], etc.
    2. Begin your response with a brief summary of which documents you're using.
    3. Throughout your response, cite specific documents using their document IDs in brackets.
    4. At the end of your response, include a 'References' section that lists all documents you cited.
    5. For each citation, explain which part of your answer used information from that document.
    """}
]
```

### 4. Ranking Visualization
The system provides detailed visualizations of how document rankings change at each stage of the retrieval process, helping users understand how the system is making decisions.

**Benefits of Ranking Visualization:**
- Provides transparency into the retrieval process
- Helps identify which ranking method performs best for different query types
- Enables debugging and fine-tuning of the retrieval pipeline

**Implementation:**
```python
# Display re-ranking visualization
with st.expander("📊 Re-ranking Analysis"):
    st.markdown("### Document Ranking Transformation")
    
    # Build comparison table
    comparison_data = []
    
    for i, reranked_doc in enumerate(retrieved_docs):
        # Find original position
        original_pos = next((j for j, d in enumerate(original_docs) if d["text"] == reranked_doc["text"]), -1)
        position_change = original_pos - i if original_pos >= 0 else "N/A"
        
        # Add to comparison data with text previews
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
```

## Implementation Details

This recipe builds on the simple RAG implementation with the following enhancements:

1. **Modular Code Structure**: The code is organized into clear functional sections for better maintainability
2. **Improved Chunking Strategy**: Smaller chunks with appropriate overlap for more precise retrieval
3. **Multi-Stage Retrieval Pipeline**:
   - Initial retrieval using vector similarity
   - Re-ranking using a cross-encoder model
   - Relevancy evaluation using LLM-based assessment
4. **Source Attribution**: Documents are assigned IDs and cited in the response
5. **Ranking Visualizations**: Interactive tables showing how document rankings change at each stage
6. **Configurable Parameters**: UI controls to adjust retrieval parameters and toggle features

## Usage

To run the advanced RAG application:

1. Install the required dependencies:
   ```bash
   pip install streamlit pandas torch qdrant-client openai llama-index sentence-transformers langchain-text-splitters tqdm docling
   ```

2. Follow the steps in the implementation files to process documents, create embeddings, and run the advanced RAG system:

   - Process files: `stage_1_process_files.py`
   - Chunk and embed: `stage_2_chunk_and_embed.py`
   - Run the advanced RAG system: `stage_3_advanced_rag_streamlit.py`

The Streamlit app provides a user-friendly interface to interact with the system, with options to:
- Configure API settings
- Select embedding and language models
- Adjust retrieval parameters
- Enable/disable re-ranking and relevancy evaluation
- View detailed visualizations of the retrieval process
- See transparent source attribution in the generated responses

## References
1. [Cross-Encoders for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html) - Information about cross-encoders for re-ranking
2. [LLM-based Relevance Evaluation](https://arxiv.org/abs/2305.14627) - Research on using LLMs to evaluate relevance
3. [Source Attribution in RAG Systems](https://towardsdatascience.com/source-attribution-in-rag-systems-9da599c76110) - Best practices for transparent source attribution
4. [Streamlit Visualization Components](https://docs.streamlit.io/library/api-reference/charts) - Tools for creating interactive visualizations
