import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from sentence_transformers import SentenceTransformer

# Function to process a CSV and split text into manageable chunks with improved parameters
def process_csv_chunking_stage(input_file, output_file, chunk_size=512, chunk_overlap=128):
    """Process CSV and split text into chunks with customizable parameters"""
    # Read input CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Validate that the required columns are present in the DataFrame
    required_columns = ['relative_path', 'text']
    if missing := [col for col in required_columns if col not in df.columns]:
        # Raise an error if required columns are missing
        raise ValueError(f"Missing required columns: {missing}")
    
    # Configure the RecursiveCharacterTextSplitter for splitting text
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,  # Specify language as Markdown
        chunk_size=chunk_size,       # Maximum size of a text chunk
        chunk_overlap=chunk_overlap, # Overlap between chunks
    )
    
    # Apply text splitter to each document's text and generate chunks
    df['chunk_list'] = df['text'].apply(
        lambda text: [s.page_content for s in text_splitter.create_documents([text])]  # Extract chunks
    )
    
    # Explode the text chunks into separate DataFrame rows
    df = df.explode('chunk_list').rename(columns={'chunk_list': 'chunk'}).reset_index(drop=True)
    
    # Save the resulting DataFrame with text chunks to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Chunking complete. Saved to {output_file}")  # Notify completion of chunking
    return df

# Function to add embeddings for text chunks and save the results
def process_csv_embedding_stage(input_file, output_file, model_name='BAAI/bge-large-zh-v1.5'):
    """Add embeddings for text chunks"""
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Initialize the SentenceTransformer embedding model with the specified name
    model = SentenceTransformer(model_name)
    
    # Convert text chunks into embeddings using the model
    chunks = df['chunk'].tolist()  # Extract the 'chunk' column as a list
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)  # Generate embeddings
    
    # Assign the computed embeddings as a new column in the DataFrame
    df['embedding'] = embeddings.tolist()
    
    # Save the updated DataFrame with embeddings to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Embedding complete. Saved to {output_file}")  # Notify completion of embedding
    return df

# Main script execution entry point
if __name__ == "__main__":
    # Configuration paths for input and output files
    input_path = "app/data/dataset_ETL1.csv"          # Input file from the previous stage
    chunk_output = "app/data/dataset_ETL2.csv"       # Output file for text chunks
    embedding_output = "app/data/dataset_ETL3.csv"   # Final output file with embeddings
    
    # Run the text chunking stage on the input data with smaller chunks for more precise retrieval
    process_csv_chunking_stage(input_path, chunk_output, chunk_size=512, chunk_overlap=128)
    
    # Run the embedding stage on the chunked text data
    process_csv_embedding_stage(chunk_output, embedding_output)
