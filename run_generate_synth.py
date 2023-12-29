import os
import scrape_arxiv
import multiprocessing
import pickle

cache_file = "datasets/arxiv_ids_cache.pkl"

# Load the cached data
with open(cache_file, "rb") as f:
    arxiv_ids = pickle.load(f)

def process_arxiv_id(arxiv_id):
    try:
        paper = scrape_arxiv.arxiv_paper(arxiv_id)
        paper.generate_summary()
        paper.generate_qa_pairs()
        paper.save_dataset_jsonl()
    except Exception as e:
        # Log the exception and arxiv_id
        print(f"Error processing {arxiv_id}: {e}")

# Create a pool of workers
pool = multiprocessing.Pool()

# Map the process_arxiv_id function to each arxiv_id in parallel
for arxiv_id in arxiv_ids:
    if not os.path.exists(f"datasets/arxiv_qa/{arxiv_id}.jsonl"):
        pool.apply_async(process_arxiv_id, args=(arxiv_id,))

# Close the pool of workers
pool.close()
pool.join()