# add summaries to the pkl files
import multiprocessing
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
import glob
import random
import pickle
import os
import argparse
import logging

# Set up logging
random_number = random.randint(10000, 99999)  # Generate a random number
log_filename = f"output/logs/{random_number}.log"  # Create a log filename with a random number
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate summaries for arXiv papers.")
parser.add_argument('--port', type=int, help='Port number for the VLLM server', required=True)
args = parser.parse_args()
vllm_port = args.port

filenames = glob.glob("input/datasets/arxiv_cache/*.pkl")
random.shuffle(filenames)

def summary(summary_input):
    inference_server_url = f"http://0.0.0.0:{vllm_port}/v1"

    llm = ChatOpenAI(
        model="input/public_models/TheBloke_Nous-Hermes-2-Yi-34B-GPTQ_gptq-4bit-32g-actorder_True/",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
	temperature=0.4,
    )

    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    return summarize_chain.run(summary_input)

def add_summary(filename):
    with open(filename, "rb") as f:
        paper = pickle.load(f)
    if "summary" in paper:
        logging.info(f"Summary already exists for {filename}")
        return
    logging.info(f"Generating summary for {filename}")
    try:
        pages = paper["pages"]
        paper["summary"] = summary(pages)
    except Exception as e:
        logging.warning(f"Error generating summary for {filename}: {e}")
        return

    with open(filename, "wb") as f:
        pickle.dump(paper, f)
        logging.info(f"Succesfully added summary to {filename}")

# Create a pool of workers
print("Creating pool of workers")   
pool = multiprocessing.Pool(16)

# Map the process_arxiv_id function to each arxiv_id in parallel
logging.info("Mapping add_summary to filenames")
for filename in filenames:
    pool.apply_async(add_summary, args=(filename,))

# Close the pool of workers
logging.info("Closing pool of workers")
pool.close()
logging.info("Waiting for workers to finish")
pool.join()
