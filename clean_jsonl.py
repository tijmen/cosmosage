
"""
Provides functions to clean and process JSONL files by removing duplicate sentences,
repeated phrases, and simple repetitions from the text fields.

Functions:
    remove_duplicate_sentences(text: str) -> str:
        Removes consecutive duplicate sentences from the given text.

    remove_repeated_phrases(text: str, min_phrase_length: int = 2, max_phrase_length: int = 22, iterations: int = 3) -> str:
        Removes repeated phrases of varying lengths from the given text.

    remove_simple_repetitions(text: str, max_len: int = 22) -> str:
        Removes simple repetitions of words or phrases up to a specified length from the given text.

    clean_text(text: str) -> str:
        Cleans the given text by removing simple repetitions, duplicate sentences, and repeated phrases.

    process_jsonl_line(json_line: str) -> str:
        Processes a single JSONL line, cleaning the "text" field if it exists.

    process_jsonl_file(input_file: str, output_file: str) -> None:
        Processes an entire JSONL file, cleaning the "text" fields in parallel using multiple processes.

Usage:
    This script can be run as a standalone program to clean a JSONL file specified by the input_file and output_file paths.

Author: Tijmen de Haan

Date: 2024-09-24    

"""

import json
import logging
import re
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count


# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def remove_duplicate_sentences(text):
    sentences = sent_tokenize(text)
    unique_sentences = []
    last_sentence = None
    for sentence in sentences:
        if sentence != last_sentence:
            unique_sentences.append(sentence)
            last_sentence = sentence
    return " ".join(unique_sentences)


def remove_repeated_phrases(
    text, min_phrase_length=2, max_phrase_length=22, iterations=3
):
    words = text.split()
    for _ in range(iterations):
        modified = False
        for phrase_length in range(max_phrase_length, min_phrase_length - 1, -1):
            i = 0
            while i <= len(words) - 2 * phrase_length:
                phrase = words[i : i + phrase_length]
                next_phrase = words[i + phrase_length : i + 2 * phrase_length]

                if phrase == next_phrase:
                    del words[i + phrase_length : i + 2 * phrase_length]
                    modified = True
                else:
                    i += phrase_length  # Skip ahead to the end of the current phrase

        # Early exit if no changes were made
        if not modified:
            break

    return " ".join(words)


def remove_simple_repetitions(text, max_len=22):
    pattern = r"\b(\w+(?:\s+\w+){0," + str(max_len - 1) + "})\s+\1\b"
    while re.search(pattern, text):
        text = re.sub(pattern, r"\1", text)
    return text


def clean_text(text):
    text = remove_simple_repetitions(text)
    text = re.sub(r"(.)\1{4,}", r"\1", text)
    text = re.sub(r"(\b\w+\b)( \1){3,}", r"\1", text)
    text = re.sub(r"(\\\\quad){2,}", "\\\\quad\\\\quad", text)
    text = re.sub(r"(\\\\qquad){2,}", "\\\\qquad\\\\qquad", text)
    text = re.sub(r"(\\\\!)+", "\\\\!", text)
    text = re.sub(r"(\[\\\\n\*\\\\\]){2,}", "[]", text)
    text = re.sub(r"(,\\\\){2,}", ",\\\\", text)
    text = re.sub(r"(cos\\\\phi\\s*){2,}", "cos\\\\phi ", text)
    text = re.sub(r"(\\\\,){2,}", "\\\\,", text)
    text = remove_duplicate_sentences(text)
    try:
        text = remove_repeated_phrases(text)
    except RecursionError:
        logging.error(
            "RecursionError occurred while removing repeated phrases. Skipping this step."
        )
    return text


def process_jsonl_line(json_line):
    try:
        data = json.loads(json_line)
        if "text" in data:
            cleaned_text = clean_text(data["text"])
            data["text"] = cleaned_text
        return json.dumps(data)
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e}")
        return None


def process_jsonl_file(input_file, output_file):
    num_processes = cpu_count()
    logging.info(f"Processing {input_file} using {num_processes} processes.")

    with open(input_file, "r") as f:
        lines = f.readlines()

    with Pool(processes=num_processes) as pool:
        cleaned_lines = pool.map(process_jsonl_line, lines)

    with open(output_file, "w") as fout:
        for cleaned_line in cleaned_lines:
            if cleaned_line is not None:
                fout.write(cleaned_line + "\n")


if __name__ == "__main__":
    base_dir = "/home/tijmen/cosmosage/datasets/cpt/"
    input_file = base_dir + "cosmosage_cpt.jsonl"
    output_file = base_dir + "cosmosage_cpt_cleaned.jsonl"
    process_jsonl_file(input_file, output_file)
