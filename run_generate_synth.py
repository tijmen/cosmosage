import os
import scrape_arxiv
import multiprocessing
import glob
import random
import pickle

# arxiv_ids = []
# for f in glob.glob("datasets/arxiv_qa/*.jsonl"):
#     arxiv_ids.append(f.split("/")[-1].split(".jsonl")[0])

with open("datasets/best_papers.pkl", "rb") as f:
    arxiv_ids = pickle.load(f)

random.shuffle(arxiv_ids)

def process_arxiv_id(arxiv_id):
    try:
        paper = scrape_arxiv.ArxivPaper(arxiv_id)
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
    if not os.path.exists(f"datasets/arxiv_qa2/{arxiv_id}.jsonl"):
        pool.apply_async(process_arxiv_id, args=(arxiv_id,))

# Close the pool of workers
pool.close()
pool.join()