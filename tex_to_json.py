#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for reading .tex files and chunking them into JSON format
for training an LLM.
"""
import os
import subprocess
import json
import pydetex.pipelines

# from pylatexenc.latex2text import LatexNodes2Text
# import TexSoup


def stats_table(tex_elements):
    """
    Calculate statistics on tex elements.
    """
    special_chars = r"\{}[]%$#@!*&^_+"
    table = []
    for string in tex_elements:
        first_20_chars = string[:20]
        length = len(string)
        backslash_count = string.count("\\")
        special_chars_count = sum(string.count(char) for char in special_chars)
        special_chars_percentage = (
            (special_chars_count / length) * 100 if length > 0 else 0
        )
        table.append(
            [
                first_20_chars,
                length,
                backslash_count,
                special_chars_count,
                special_chars_percentage,
            ]
        )
    return table


def filter_for_good_elements(tex_elements):
    '''
    Delete any elements that have more than 2% special characters.
    '''
    stats = stats_table(tex_elements)
    return [element for element, stat in zip(tex_elements, stats) if stat[4] < 2]


def parse_tex_file(file_path):
    """
    This function takes a file path to a .tex file as input, reads the file,
    converts the LaTeX content to plain text, and splits the text into paragraphs.

    Args:
        file_path (str): The path to the .tex file.

    Returns:
        list: A list of paragraphs from the .tex file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tex = f.read()

    if not tex:
        return []

    # Check if the string is not empty before passing it to pydetex
    try:
        datablocks = pydetex.pipelines.simple(tex).split("\n\n")
    except Exception as e:
        print(f"Error parsing tex file {file_path}: {e}")
        datablocks = []

    good_datablocks = filter_for_good_elements(datablocks)

    return good_datablocks

    # Previous attempt using TexSoup

    # # Extract abstract, captions, sections, subsections, subsubsections,
    # # and subsubsubsections from LaTeX to a list of strings
    # def strip_latex_commands(text):
    #     flattenend_text = ""
    #     for line in text:
    #         if not line.startswith("%"):
    #             flattenend_text += line + "\n"
    #     return flattenend_text

    # soup = TexSoup.TexSoup(tex)
    # tex_elements = []
    # for abstract in soup.find_all("abstract"):
    #     tex_elements.append(strip_latex_commands(abstract.text))
    # for caption in soup.find_all("caption"):
    #     tex_elements.append(strip_latex_commands(caption.text))
    # for section in soup.find_all(
    #     ["section", "subsection", "subsubsection", "subsubsubsection"]
    # ):
    #     # name = section.name
    #     # text = '\n'.join(section.text)
    #     # tex_elements.append(f"{name}\n{text}\n")
    #     # Section heading
    #     heading = f"{section.name}\n{section.text}"

    #     # Body text
    #     body = []
    #     current = section
    #     while True:
    #         next_siblings = current.next_siblings
    #         if not next_siblings:
    #             break
    #         if isinstance(next_siblings[0], TexSoup.TexNode):
    #             # Break if next sectioning command
    #             if next_siblings[0].name in [
    #                 "section",
    #                 "subsection",
    #                 "subsubsection",
    #                 "subsubsubsection",
    #             ]:
    #                 break
    #         # Append text from sibling
    #         body.extend(next_siblings[0].text)
    #         current = next_siblings[0]

    #         body = "\n".join(body)

    #         tex_elements.append(f"{heading}\n{body}\n")

    # return tex_elements


def parse_tex_files(folder_path):
    """
    This function takes a folder path as input, reads all .tex files in the folder,
    converts the LaTeX content to plain text, and splits the text into paragraphs.

    Args:
        folder_path (str): The path to the folder containing .tex files.

    Returns:
        dict: A dictionary where the keys are the filenames and the values are lists
        of paragraphs from the .tex files.
    """
    output = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".tex"):
            file_path = os.path.join(folder_path, filename)
            output[filename] = parse_tex_file(file_path)

    return output


def save_to_json(data, output_file):
    """
    This function takes a dictionary and a file path as input, and writes the dictionary
    to the file in JSON format.

    Args:
        data (dict): The data to be written to the file.
        output_file (str): The path to the file where the data will be written.
    """
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

def load_from_json(input_file):
    """
    This function takes a file path as input, and loads the JSON data from the file.

    Args:
        input_file (str): The path to the file where the data will be read.

    Returns:
        dict: The data from the file.
    """
    with open(input_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data

def detex_files(folder):
    for file in os.listdir(folder):
        if file.endswith(".tex"):
            detex_file = os.path.splitext(file)[0] + ".detex"
            detex_path = os.path.join(folder, detex_file)
            
            # Check if the .detex file already exists
            if not os.path.exists(detex_path):
                tex_path = os.path.join(folder, file)

                # Run detex command and redirect output to .detex file
                with open(detex_path, "w") as output:
                    subprocess.run(["detex", tex_path], stdout=output)

if __name__ == "__main__":
    tex_files_path = "tex_files/"
    json_file_path = "output/arxiv_tex.json"

    detex_files(tex_files_path)

    # # Parse .tex files
    # files = parse_tex_files(tex_files_path)

    # # Save parsed files to JSON
    # save_to_json(files, json_file_path)
