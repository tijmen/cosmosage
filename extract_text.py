import os
import glob
import subprocess
import platform
from multiprocessing import Pool
import edspdf
from pathlib import Path

def extract_text_from_pdf(path):
    try:
        # Initialize the EDS-PDF pipeline
        model = edspdf.Pipeline()
        model.add_pipe("mupdf-extractor")
        model.add_pipe("simple-aggregator", name="aggregator",config={
                "new_line_threshold": 0.2,
                "new_paragraph_threshold": 1.5,
                "label_map": {"body": "text","table": "text",}})
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

        return output.decode("utf-8", errors='ignore')

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
    output_dir = "textbooks_extracted"
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


if __name__ == "__main__":
    textbooks_dir = "astro_textbooks/"  # or the path to your textbooks directory
    process_textbooks_multiprocess(textbooks_dir)
