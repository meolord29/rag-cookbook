Types of questions this recipe answers:
- How do I put my data into my app?
    - How do I ingest data into my app?
    - How do I "embed" documents?
    - How do I set up a vector database?
    - How do I incorporate a vector database into the application?

1. https://github.com/docling-project/docling/tree/main/docs  
        - Information about Docling, how to parse documents  
2. https://huggingface.co/BAAI/bge-large-en-v1.5  
        - Straightforward small embedding model, fairly good as well  
3. https://sbert.net/  
        - The Python package for quick embedding functionality, super simple to use  
4. https://qdrant.tech/documentation/
        - Great vector database that can not only run in memory but is feature-rich, and can easily run on a server or cloud without changing any of the code.  
5. https://dev.to/rutamstwt/langchain-document-splitting-21im  
        - Recursive text splitting with respect to markdown annotations  

```bash
pip install qdrant-client sentence-transformers tqdm pandas docling langchain-text-splitters streamlit openai
```  