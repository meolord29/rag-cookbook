#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs("app/data", exist_ok=True)
    os.makedirs("app/folder_for_files", exist_ok=True)
    print("✅ Created data directories")

def run_stage_1():
    """Run the file processing stage"""
    print("\n🔄 Running Stage 1: Processing Files...")
    subprocess.run([sys.executable, "app/stage_1_process_files.py"])

def run_stage_2():
    """Run the chunking and embedding stage"""
    print("\n🔄 Running Stage 2: Chunking and Embedding...")
    subprocess.run([sys.executable, "app/stage_2_chunk_and_embed.py"])

def run_stage_3():
    """Run the Streamlit app"""
    print("\n🚀 Launching Advanced RAG Streamlit App...")
    subprocess.run(["streamlit", "run", "app/stage_3_advanced_rag_streamlit.py"])

def main():
    parser = argparse.ArgumentParser(description="Run the Advanced RAG system")
    parser.add_argument(
        "--stage", 
        type=int, 
        choices=[1, 2, 3], 
        help="Run a specific stage (1, 2, or 3)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all stages sequentially"
    )
    
    args = parser.parse_args()
    
    create_data_directory()
    
    if args.all:
        run_stage_1()
        run_stage_2()
        run_stage_3()
    elif args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
