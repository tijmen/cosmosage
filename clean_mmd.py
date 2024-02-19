import glob
import json
import os
import random
import re
import logging
from multiprocessing import Pool, cpu_count
from nltk.tokenize import sent_tokenize
from collections import deque

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_duplicate_sentences(text):
    sentences = sent_tokenize(text)
    unique_sentences = []
    last_sentence = None
    for sentence in sentences:
        if sentence != last_sentence:
            unique_sentences.append(sentence)
            last_sentence = sentence
    return ' '.join(unique_sentences)

def remove_repeated_phrases(text, min_phrase_length=2, max_phrase_length=22, iterations=3):
    words = text.split()
    for _ in range(iterations):
        modified = False
        for phrase_length in range(max_phrase_length, min_phrase_length - 1, -1):
            i = 0
            while i <= len(words) - 2 * phrase_length:
                phrase = words[i:i + phrase_length]
                next_phrase = words[i + phrase_length:i + 2 * phrase_length]

                if phrase == next_phrase:
                    del words[i + phrase_length:i + 2 * phrase_length]
                    modified = True
                else:
                    i += phrase_length  # Skip ahead to the end of the current phrase

        # Early exit if no changes were made
        if not modified:
            break

    return ' '.join(words)


def remove_simple_repetitions(text, max_len=22):
    # Find all sequences of words up to max_len that are repeated immediately
    pattern = r'\b(\w+(?:\s+\w+){0,' + str(max_len-1) + '})\s+\1\b'
    # Replace repeated sequences with a single occurrence
    while re.search(pattern, text):
        text = re.sub(pattern, r'\1', text)
    return text

def clean_text(text):
    text = remove_simple_repetitions(text)
    text = re.sub(r'(.)\1{4,}', r'\1', text)  # Remove repeated characters more than 4 times
    text = re.sub(r'(\b\w+\b)( \1){3,}', r'\1', text)  # Remove repeated words more than 3 times consecutively
    text = re.sub(r'(\\\\quad){2,}', '\\\\quad\\\\quad', text)  # Reduce repeated '\\quad' sequences to two instances
    text = re.sub(r'(\\\\qquad){2,}', '\\\\qquad\\\\qquad', text)  # Reduce repeated '\\qquad' sequences to two instances
    text = re.sub(r'(\\\\!)+', '\\\\!', text)  # Simplify repeated '\\!' sequences to a single instance
    text = re.sub(r'(\[\\\\n\*\\\\\]){2,}', '[]', text)  # Simplify repeated '[\\n...]' patterns
    text = re.sub(r'(,\\\\){2,}', ',\\\\', text)  # Simplify repeated ',\\' sequences to a single instance
    text = re.sub(r'(cos\\\\phi\\s*){2,}', 'cos\\\\phi ', text)  # Simplify repeated 'cos\\phi' sequences to a single instance with a trailing space
    text = re.sub(r'(\\\\,){2,}', '\\\\,', text)  # Corrected to reduce repeated '\\,' sequences to a single instance
    text = remove_duplicate_sentences(text)
    try:
        text = remove_repeated_phrases(text)
    except RecursionError:
        logging.error("RecursionError occurred while removing repeated phrases. Skipping the removal of repeated phrases.")
    # Add more cleaning rules as needed
    return text


def process_file(filename):
    try:
        logging.info(f"Processing file {filename}")
        with open(filename, "r") as f:
            text = f.read()
            cleaned_text = clean_text(text)
            json_line = json.dumps({"text": cleaned_text})
        return json_line
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        return None

def process_directory(directory, outfile_path):
    filenames = glob.glob(os.path.join(directory, "*.mmd"))
    random.shuffle(filenames)
    num_processes = cpu_count() #min(cpu_count(), 8)
    logging.info(f"Processing files in {directory} using {num_processes} processes.")

    with Pool(processes=num_processes) as pool:
        json_lines = pool.map(process_file, filenames)
        with open(outfile_path, "w") as fout:
            for json_line in json_lines:
                if json_line is not None:
                    fout.write(json_line + '\n')

if __name__ == "__main__":
   directories = ["datasets/arxiv_mmd", "datasets/textbooks_mmd"]
   output_files = ["datasets/arxiv_mmd.jsonl", "datasets/textbooks_mmd.jsonl"]

   for directory, outfile in zip(directories, output_files):
       process_directory(directory, outfile)

