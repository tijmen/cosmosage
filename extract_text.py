import os
import glob
import subprocess
from multiprocessing import Pool
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


def extract_text_from_pdf(path):
    try:
        with open(path, "rb") as fp:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()
            for page in PDFPage.get_pages(
                fp,
                pagenos,
                maxpages=maxpages,
                password=password,
                caching=caching,
                check_extractable=True,
            ):
                interpreter.process_page(page)
            text = retstr.getvalue()
            device.close()
            retstr.close()
            return text
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def extract_text_from_djvu(file_path):
    try:
        output = subprocess.check_output(
            ["C:\Program Files (x86)\DjVuLibre\djvutxt", file_path]
        )
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
    with Pool(6) as pool:
        pool.map(process_single_book, books_to_process)


if __name__ == "__main__":
    textbooks_dir = "./"  # or the path to your textbooks directory
    process_textbooks_multiprocess(textbooks_dir)
