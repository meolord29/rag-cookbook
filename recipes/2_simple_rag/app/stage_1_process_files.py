import os
import pandas as pd
import hashlib
from tqdm import tqdm
from docling.document_converter import DocumentConverter

# Function to compute SHA256 hash of a file, with exception handling for errors
def compute_sha256_hash(file_path):
    """Compute SHA256 hash of a file with error handling"""
    try:
        # Initialize the SHA256 hash object
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as file:
            # Read the file in chunks and update the hash
            while chunk := file.read(8192):  # Read up to 8192 bytes at once
                sha256_hash.update(chunk)
        # Return the computed hash as a hexadecimal string
        return sha256_hash.hexdigest()
    except Exception as e:
        # Return an error message if hash calculation fails
        return f"Hash Error: {str(e)}"

# Function to traverse a directory and collect metadata about files
def get_file_metadata(root_dir):
    """Collect file paths and basic metadata"""
    file_data = []  # Initialize list to store file metadata
    
    # Walk through the directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:  # Iterate through files in each directory
            # Construct full and relative paths for current file
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            # Extract filename and extension
            file_name, file_ext = os.path.splitext(filename)
            
            # Append metadata about the current file
            file_data.append({
                'full_path': full_path,
                'relative_path': rel_path,
                'file_name': filename,
                'file_type': file_ext[1:].lower()  # Remove the dot from file extension
            })
    # Return the list of file metadata
    return file_data

# Function to process files: compute hashes and extract text content
def process_files(file_data):
    """Process files to extract content and compute hashes"""
    # Initialize the document converter (used for extracting text)
    converter = DocumentConverter()
    
    # Iterate through all files with a progress bar for tracking
    for item in tqdm(file_data, desc="Processing files"):
        # Compute SHA256 hash for the current file
        item['sha256_hash'] = compute_sha256_hash(item['full_path'])
        
        # Extract text content using the converter
        try:
            result = converter.convert(item['full_path'])
            # Export the document content to markdown format
            item['text'] = result.document.export_to_markdown()
        except Exception as e:
            # Log an error message if text extraction fails
            item['text'] = f"Extraction Error: {str(e)}"
    
    # Return the file data with added content and hash information
    return file_data

# Main function that orchestrates file processing workflow
def process_folder(root_dir):
    """Main processing workflow"""
    # Phase 1: Collect metadata for all files in the directory
    files = get_file_metadata(root_dir)
    
    # Phase 2: Process files to extract content and compute hashes
    processed_files = process_files(files)
    
    # Create a DataFrame from the processed file data
    df = pd.DataFrame(processed_files).drop(columns=['full_path'])  # Remove 'full_path' column
    
    # Add a column indicating whether the file is a duplicate based on its hash value
    df['is_duplicate'] = df.duplicated(subset=['sha256_hash'], keep=False)
    
    # Return the final DataFrame
    return df

# Example entry point for script execution
if __name__ == "__main__":
    # Specify the root directory containing files to be processed
    root_directory = "app/folder_for_files"
    
    # Process the files in the directory and generate a DataFrame
    df = process_folder(root_directory)
    
    # Display results: number of processed files and duplicates found
    print(f"\nProcessed {len(df)} files. Duplicates found: {df['is_duplicate'].sum()}")
    print(df.sort_values('is_duplicate', ascending=False).head(10))  # Show top 10 results (duplicates first)
    
    # Optional: Detailed analysis of duplicates
    print("\nDuplicate files analysis:")
    # Count occurrences of duplicate files based on relative path and hash
    print(df[df['is_duplicate']][['relative_path', 'sha256_hash']].value_counts())
    
    # Save the processed DataFrame to a CSV file
    df.to_csv("app/data/dataset_ETL1.csv")