import os
import glob
import subprocess
import platform
import re
from multiprocessing import Pool
import edspdf
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random


def extract_text_from_pdf(path):
    try:
        # Initialize the EDS-PDF pipeline
        model = edspdf.Pipeline()
        model.add_pipe("mupdf-extractor")
        model.add_pipe(
            "simple-aggregator",
            name="aggregator",
            config={
                "new_line_threshold": 0.2,
                "new_paragraph_threshold": 1.5,
                "label_map": {
                    "body": "text",
                    "table": "text",
                },
            },
        )
        # Read the PDF file
        pdf = Path(path).read_bytes()

        # Apply the pipeline
        processed_pdf = model(pdf)

        # Check if 'body' key exists in aggregated_texts
        if "body" in processed_pdf.aggregated_texts:
            text = processed_pdf.aggregated_texts["body"].text
        else:
            # Extract and concatenate text from each TextBox
            text = "\n".join([box.text for box in processed_pdf.lines])

        return text
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def extract_text_from_djvu(file_path):
    try:
        # Determine the path of djvutxt based on the operating system
        if platform.system() == "Windows":
            djvutxt_path = r"C:\Program Files (x86)\DjVuLibre\djvutxt"
        else:  # Assuming Linux
            djvutxt_path = "djvutxt"  # Typically just the command name on Linux

        # Use a list for the command and its arguments
        command = [djvutxt_path, file_path]
        output = subprocess.check_output(command)

        return output.decode("utf-8", errors="ignore")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_text_from_file(file_path, file_extension):
    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".djvu":
        return extract_text_from_djvu(file_path)
    else:
        return None


def process_single_book(file_path):
    file_key = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]
    output_dir = "datasets/textbooks_extracted"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{file_key}.txt")

    if not os.path.isfile(output_file_path):
        print(f"Starting processing {file_path}")
        text = extract_text_from_file(file_path, file_extension)
        if text is not None:
            with open(output_file_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(text)
            print(f"Processed {file_path}")
        else:
            # write empty file to indicate that we tried to process it
            with open(output_file_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write("")


def find_books_to_process(textbooks_dir):
    all_files = glob.glob(
        os.path.join(textbooks_dir, "**", "*.pdf"), recursive=True
    ) + glob.glob(os.path.join(textbooks_dir, "**", "*.djvu"), recursive=True)
    to_process = []
    for file_path in all_files:
        file_key = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join("textbooks_extracted", f"{file_key}.txt")
        if not os.path.isfile(output_file_path):
            to_process.append(file_path)
    return to_process


def process_textbooks_multiprocess(textbooks_dir):
    books_to_process = find_books_to_process(textbooks_dir)
    with Pool() as pool:
        pool.map(process_single_book, books_to_process)


def preprocess_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Remove hyphenation
    text = re.sub(r"-\n", "", text)

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Remove long sequences of special characters and numbers
    text = re.sub(r"[\W\d_]{10,}", "", text)

    # Replace sequences of newline + few-digit number + newline with paragraph token
    text = re.sub(r"\n\d{1,3}\n", " <PARA> ", text)

    # Replace double line breaks with a special token
    text = text.replace("\n\n", " <PARA> ")

    # Replace single line breaks with space
    text = text.replace("\n", " ")

    # Split into paragraphs and further split each paragraph into sentences
    paragraphs = [para.strip() for para in text.split(" <PARA> ") if para.strip()]

    return paragraphs


def calculate_percentages(para):
    total_chars = len(para)
    char_classes = {
        "spaces": " ",
        "digits": "0123456789",
        "capital_letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "lowercase_letters": "abcdefghijklmnopqrstuvwxyz",
        "newlines": "\n",
        "backslashes": "\\",
        "periods": ".",
        "exclamation_marks": "!",
        "question_marks": "?",
    }

    percentages = {}
    for class_name, chars in char_classes.items():
        count = sum(para.count(char) for char in chars)
        percentages[class_name] = 100 * count / total_chars if total_chars > 0 else 0

    return percentages


def histogram_percentages(books_paragraphs):
    percentage_data = {key: [] for key in calculate_percentages("").keys()}

    for book in books_paragraphs:
        for para in book:
            percentages = calculate_percentages(para)
            for class_name, percent in percentages.items():
                percentage_data[class_name].append(percent)

    plt.figure()
    for class_name, data in percentage_data.items():
        plt.hist(data, bins=50, alpha=0.5, label=class_name.capitalize())
    plt.legend()
    plt.xlabel("Percentage")
    plt.ylabel("Number of Paragraphs")
    plt.title("Character Distribution")
    plt.show()


def is_paragraph_good(para_percentages, bounds):
    for class_name, (lower_bound, upper_bound) in bounds.items():
        if not (lower_bound <= para_percentages[class_name] <= upper_bound):
            return False
    return True


def filter_paragraphs(book, bounds):
    good_paragraphs = [
        para for para in book if is_paragraph_good(calculate_percentages(para), bounds)
    ]
    return good_paragraphs


def filter_textbooks(books_paragraphs, bounds):
    filtered_books = []
    for book in books_paragraphs:
        good_paragraphs = filter_paragraphs(book, bounds)
        if len(good_paragraphs) >= len(book) / 2:  # Majority of paragraphs are good
            filtered_books.append(good_paragraphs)
    return filtered_books


def textbooks_to_jsonl(output_file_path):
    """
    Primary function to convert the textbooks to JSONL format.
    """
    # Parse the books into individual .txt files
    process_textbooks_multiprocess("datasets/astro_textbooks/")
    process_textbooks_multiprocess("datasets/physics_textbooks/")

    # Preprocess, to get a list of books, where each book is a list of paragraphs
    book_paths = glob.glob("datasets/textbooks_extracted/*.txt")
    books_paragraphs = [preprocess_text(file_path) for file_path in book_paths]

    # Mark each paragraph as good or bad based on whether the rate of certain
    # characters is within the distribution
    # histogram_percentages(books_paragraphs)
    bounds = {
        "spaces": (6, 24),
        "digits": (0, 15),
        "capital_letters": (1, 23),
        "lowercase_letters": (50, 95),
        "newlines": (0, 5),
        "backslashes": (0, 5),
        "periods": (0, 8),
        "exclamation_marks": (0, 5),
        "question_marks": (0, 6),
    }

    # Filter the textbooks
    filtered_books_paragraphs = filter_textbooks(books_paragraphs, bounds)

    # save to JSON: full, training, and evaluation sets
    root = "/home/tijmen/cosmosage/datasets/"

    # Collect all paragraphs
    all_paragraphs = []
    for book in filtered_books_paragraphs:
        for para in book:
            all_paragraphs.append({"text": para})

    # Make a flat JSONL that has one entry per book
    textbooks = []
    for book in filtered_books_paragraphs:
        if len(book) > 100:
            textbooks.append("\n\n".join(book))

    random.shuffle(textbooks)

    with open(output_file_path, "w", encoding="utf-8") as flat_f:
        for textbook in textbooks:
            flat_f.write(json.dumps({"text": textbook}) + "\n")


# if __name__ == "__main__":
#     textbooks_to_jsonl("datasets/textbooks.jsonl")

"""
Converting the textbooks to raw text for continued pretraining is one approach.

However, there's an alternative approach. Like with the arXiv papers, we can turn the textbooks
into QA pairs instead.

The following is code is an attempt to make such QA pairs.
"""

import os
import re
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from itertools import chain
from multiprocessing import Pool


def generate_qa_pair(args):
    text, title, author = args

    class question_answer(BaseModel):
        question: str = Field(..., description="Question framed.")
        answer: str = Field(..., description="Answer to the question.")

    class output(BaseModel):
        output: list[question_answer] = []

    # connect to the VLLM server that I started separately with something like
    #  python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.2
    inference_server_url = "http://0.0.0.0:8000/v1"

    llm = ChatOpenAI(
        # model="mistralai/Mistral-7B-Instruct-v0.2",
        model="/home/tijmen/public_models/TheBloke_Mixtral-8x7B-Instruct-v0.1-GPTQ_gptq-4bit-32g-actorder_True",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=4096,
        temperature=0.4,
    )

    parser = PydanticOutputParser(pydantic_object=output)

    prompt = (
        """You are an expert on cosmology and are tasked with generating questions and answers. You make question-answer pairs from a given PASSAGE of a cosmology textbook. The questions contain the context and can be understood by themselves. DO NOT reference the PASSAGE itself. The answer should be long and demonstrate an excellent understanding of the subject matter.
    Textbook: """
        + title
        + """
    Author: """
        + author
        + """
    PASSAGE: {text}

    {format_instructions}

    Output:"""
    )

    _prompt = PromptTemplate(
        template=prompt,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = _prompt.format_prompt(text=text)
    message = [HumanMessage(content=_input.to_string())]

    llm_response = llm(message).content

    # Check if the response is not empty or None
    if not llm_response:
        print("The response from the LLM is empty or None.")
        return []

    # try:
    #     greedy = True
    #     if not greedy:
    #         json_result = json.loads(llm_response)
    #     else:

    # except json.decoder.JSONDecodeError as e:
    #     print("Cannot serialize this output because of JSONDecodeError:", e)
    #     print("Original content causing error:", llm_response)
    #     print("=================")
    #     return []

    # output_list = []

    # # Ensure json_result is a list for uniform processing
    # json_result = [json_result] if not isinstance(json_result, list) else json_result

    # for item in json_result:
    #     if isinstance(item, dict):
    #         if "question" in item and "answer" in item:
    #             # Append the item containing 'question' and 'answer'
    #             output_list.append(item)
    #         elif "output" in item:
    #             # Handle 'output' key, ensuring it's iterable
    #             if isinstance(item["output"], list):
    #                 output_list.extend(item["output"])
    #             else:
    #                 print(
    #                     f"Expected a list for 'output', but got a different type: {item['output']}"
    #                 )
    #         else:
    #             print(f"JSON item format not as expected: {item}")
    #     else:
    #         print(f"JSON item is not a dictionary: {item}")

    def preprocess_string(s):
        # Escape special characters and handle multiline strings
        return s.replace("\n", "\\n").replace('"', '\\"')

    def extract_qa_pairs(text):
        qa_pairs = []

        def match_patterns(text):
            # Multiple patterns to account for different structures
            patterns = [
                r'\{\s*"question":\s*"(.*?)"\s*,\s*"answer":\s*"(.*?)"\s*\}',  # Original pattern
                r"\"question\":\s*\"(.*?)\"\s*,\s*\"answer\":\s*\"(.*?)\"",  # Pattern for nested within an array
                r'[{[]\s*\\?"question\\?":\s*\\?"(.*?)\\?"\s*,\s*\\?"answer\\?":\s*\\?"(.*?)\\?"\s*[}\]]',  # Pattern with escaped quotes
                r'\\?"question\\?":\s*\\?"(.*?)\\?"\s*,\s*\\?"answer\\?":\s*\\?"(.*?)\\?"(?=,\s*\\?[{[]|\s*\\?]\])',  # Pattern for multiple JSON objects in an array
                r'[{[]\s*\\?"output\\?":\s*\[\s*{.*?"question":\s*\'(.*?)\'\s*,\s*"answer":\s*\'(.*?)\'\s*}(?:,\s*{.*?}|])',  # Pattern for nested structure with single quotes
                r'"question":\s*"(.+?)"\s*,\s*"answer":\s*"((?:[^"]|"(?![},]))+)"',  # New pattern to capture multi-sentence answers
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            # If no pattern matches
            print(f"No matches found for patterns in text: {text}")
            return None

        matches = match_patterns(text)
        if not matches:
            return []

        for question, answer in matches:
            try:
                # Manually construct the dictionary from question and answer
                qa_pair = {
                    "question": question.replace("\n", " ").replace('\\"', '"'),
                    "answer": answer.replace("\n", " ").replace('\\"', '"'),
                }
                qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"Error constructing QA pair: {e}")
                print("Failed match:", question, answer)

        return qa_pairs

    output_list = extract_qa_pairs(llm_response)
    return output_list


class TextBook:
    """
    Class to hold a textbook and its metadata.
    Can generate QA pairs with an LLM.
    """

    def __init__(self, filepath):
        self.filepath = filepath

        # Extract title, author, year from filepath
        #   format is "title, by author, year.txt"
        base = os.path.splitext(os.path.basename(self.filepath))[0]
        match = re.match(r"(.+), by (.+?), (\d{4})", base)
        if match:
            self.title, self.author, self.year = match.groups()
        else:
            raise ValueError(f"Could not parse {self.filepath}")

        self.qa_pairs = []
        self.load_text()

    def load_text(self):
        with open(self.filepath, "r") as f:
            self.text = f.read()

    def generate_qa_pairs(self, multiprocess=True):
        def chunk_text(text, chunk_size=1524, overlap=500):
            """Divide the text into overlapping chunks."""
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)
            ]

        def create_qa_pairs(text_chunks, title, author):
            """Generate QA pairs for each chunk of text."""
            with Pool() as pool:
                result = pool.map(
                    generate_qa_pair, [(chunk, title, author) for chunk in text_chunks]
                )
            return list(chain.from_iterable(result))

        # Prepare chunks of text with overlap
        text_chunks = chunk_text(self.text)
        self.qa_pairs = []

        if multiprocess:
            # Generate QA pairs using multiprocessing
            self.qa_pairs = create_qa_pairs(text_chunks, self.title, self.author)
        else:
            for chunk in text_chunks:
                qa_pairs = generate_qa_pair(chunk, self.title, self.author)
                self.qa_pairs.extend(qa_pairs)

    def save_dataset_jsonl(self):
        with open(f"datasets/cosmology_textbooks_qa/{self.author}.jsonl", "w") as f:
            for item in self.qa_pairs:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import glob

    textbooks = []
    for filepath in glob.glob("datasets/cosmology_textbooks/*.txt"):
        textbooks.append(TextBook(filepath))
    for textbook in textbooks:
        textbook.generate_qa_pairs(multiprocess=True)
        textbook.save_dataset_jsonl()
        print(f"Saved {textbook.author} to jsonl")
