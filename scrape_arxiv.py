#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for scraping the arXiv. Uses arxiv-sanity-lite as a library.
"""
import io
import tarfile
import os
import requests

# arxiv.py used as a library
# downloaded like a static library from arxiv-sanity-lite github master branch 20231121
# from arxiv import get_response, parse_response


def extract_tex(papers, tex_files_path):
    """
    Download papers from the arxiv. It downloads the tar file, extracts the .tex
    file, and saves it to disk.

    Args:
        papers (list): A list of paper IDs.
        tex_files_path (str): The path to the folder where the .tex files will be saved.

    Returns:
        None
    """
    for paper in papers:
        output_file_path = os.path.join(tex_files_path, paper + ".tex")
        if os.path.exists(output_file_path):
            continue

        # fetch paper from arxiv
        url = "https://arxiv.org/e-print/" + paper
        # resp = get_response(search_query=paper, start_index=1)

        # send GET request to the URL of the paper
        response = requests.get(url, timeout=10)
        tar_bytes = response.content

        # extract tarball, retain .tex file and delete the rest
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
                tex_file_name = None
                # iterate over tar.getmembers() to find the one that ends in .tex
                for member in tar.getmembers():
                    if member.name.endswith(".tex"):
                        tex_file_name = member.name
                        break

                if tex_file_name:
                    tex_file = tar.extractfile(tex_file_name)
                    try:
                        tex_content = tex_file.read().decode("utf-8")
                    except Exception as e:
                        print(f"Error reading tex file for paper {paper}: {e}")
                        continue
                else:
                    print(f"No .tex file found in the tar archive for paper {paper}")
                    continue

            del tar_bytes
        except Exception as e:
            print(f"Error extracting tar archive for paper {paper}: {e}")
            continue

        # now save the tex file to disk
        if tex_file_name:
            with open(
                os.path.join(tex_files_path, paper + ".tex"), "w", encoding="utf-8"
            ) as file:
                file.write(tex_content)
