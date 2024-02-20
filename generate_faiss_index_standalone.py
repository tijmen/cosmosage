"""
Take an argument 0-7
Load input/mmd_cache_chunk[0-9].pkl
Create FAISS index
"""

import os
import argparse 
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

parser = argparse.ArgumentParser(description="Create a FAISS index from a chunk of the MMD cache")
parser.add_argument("chunk", type=int, help="The chunk number (0-7)")
args = parser.parse_args()
chunk = args.chunk

in_file = f"input/mmd_cache_chunk/mmd_cache_chunk{chunk}.pkl"
out_file = f"output/mmd_cache_chunk/faiss_index_chunk{chunk}.bin"

with open(in_file, "rb") as f:
    chunked_docs = pickle.load(f)
index = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5'))
index.save_local(out_file)