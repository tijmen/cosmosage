import sqlite3
import pandas as pd
import zlib
import pickle

def decompress_value(value):
    try:
        return zlib.decompress(value)
    except zlib.error as e:
        return f"Error decompressing: {e}"

def deserialize_value(value):
    try:
        return pickle.loads(value)
    except pickle.UnpicklingError as e:
        return f"Error deserializing: {e}"

def count_unique_tags(deserialized_value):
    if isinstance(deserialized_value, dict):
        return len(deserialized_value)
    else:
        return 0

def count_total_elements(deserialized_value):
    if isinstance(deserialized_value, dict):
        total_elements = sum(len(papers) for papers in deserialized_value.values())
        return total_elements
    else:
        return 0

def extract_user_tag_data(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read all entries from the 'tags' table
    all_tags = pd.read_sql_query("SELECT * FROM tags;", conn)

    # Close the database connection
    conn.close()

    # Apply decompression and deserialization
    all_tags['decompressed_value'] = all_tags['value'].apply(decompress_value)
    all_tags['deserialized_value'] = all_tags['decompressed_value'].apply(deserialize_value)

    # Count the number of unique tags and total elements for each user
    all_tags['num_unique_tags'] = all_tags['deserialized_value'].apply(count_unique_tags)
    all_tags['total_elements'] = all_tags['deserialized_value'].apply(count_total_elements)

    # Create a summary table of users, their number of unique tag types, and total number of tags
    user_tag_summary = all_tags.groupby('key').agg({
        'num_unique_tags': 'sum',
        'total_elements': 'sum'
    }).reset_index()

    user_tag_summary.rename(columns={'key': 'User', 'num_unique_tags': 'Number of Tag Types', 'total_elements': 'Number of Tags'}, inplace=True)

    return user_tag_summary


def extract_arxiv_numbers(deserialized_value):
    arxiv_numbers = set()
    if isinstance(deserialized_value, dict):
        for papers in deserialized_value.values():
            arxiv_numbers.update(papers)
    return arxiv_numbers

def extract_unique_arxiv_numbers(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read all entries from the 'tags' table
    all_tags = pd.read_sql_query("SELECT * FROM tags;", conn)

    # Close the database connection
    conn.close()

    # Apply decompression and deserialization
    all_tags['decompressed_value'] = all_tags['value'].apply(decompress_value)
    all_tags['deserialized_value'] = all_tags['decompressed_value'].apply(deserialize_value)

    # Extract arxiv numbers for all unique tagged papers
    all_tags['arxiv_numbers'] = all_tags['deserialized_value'].apply(extract_arxiv_numbers)

    # Create a set to hold all unique arxiv numbers
    unique_arxiv_numbers = set()
    for numbers in all_tags['arxiv_numbers']:
        unique_arxiv_numbers.update(numbers)

    return unique_arxiv_numbers

if __name__=='__main__':
    db_path = 'dict_20231123.db' 
    user_tag_summary = extract_user_tag_data(db_path)
    print(user_tag_summary) 