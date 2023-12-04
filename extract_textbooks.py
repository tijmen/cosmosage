import os
import glob
import subprocess
import platform
import re
from multiprocessing import Pool
import edspdf
from pathlib import Path
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    textbooks_dir = "astro_textbooks/"  # or the path to your textbooks directory
    process_textbooks_multiprocess(textbooks_dir)
