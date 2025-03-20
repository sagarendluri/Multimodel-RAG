import tabula
import faiss
import json
import base64
import pymupdf
import requests
import os
import logging
import numpy as np
import warnings
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from IPython import display

# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

# Process tables
def process_tables(filepath, doc, page_num, base_dir, items):
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
            table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
            with open(table_file_name, 'w') as f:
                f.write(table_text)
            items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})
    except Exception as e:
        print(f"Error extracting tables from page {page_num}: {str(e)}")
# Process text chunks
def process_text_chunks(filepath,text, text_splitter, page_num, base_dir, items):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

# Process images
def process_images(filepath, page, page_num, base_dir, items):
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"
        pix.save(image_name)
        with open(image_name, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf8')
        items.append({"page": page_num, "type": "image", "path": image_name, "image": encoded_image})

# Process page images
def process_page_images(filepath, page, page_num, base_dir, items):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

from pathlib import Path

def list_files_with_extension(folder_path, extension):
    """
    Lists all files with a given extension in a folder.

    Args:
        folder_path: Path to the folder.
        extension: The file extension (e.g., ".pdf", ".txt").

    Returns:
        A list of file names with the specified extension.
    """

    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        return []  # Return an empty list if the folder doesn't exist

    file_list = [file.name for file in folder_path.glob(f"*{extension}")]
    return file_list

# Example usage:
folder_path = "data_pdf"  # Replace with the path to your folder
extension = ".pdf"
pdf_files = list_files_with_extension(folder_path, extension)
os.makedirs("data", exist_ok=True)
items = []
for pdf_path_str in pdf_files:
    pdf_path = os.path.join("data", pdf_path_str)
    # pdf_path = Path(pdf_path_str)
    # # if not pdf_path.exists():
    #     print(f"PDF file not found: {pdf_path}")
    #     continue
    try:
        doc = pymupdf.open(pdf_path)
        num_pages = len(doc)
        base_dir = "data"

        # Creating the directories
        create_directories(base_dir)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
        items = []

        # Process each page of the PDF
        for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
            page = doc[page_num]
            text = page.get_text()
            process_tables(pdf_path, doc, page_num, base_dir, items)
            process_text_chunks(pdf_path, text, text_splitter, page_num, base_dir, items)
            process_images(pdf_path, page, page_num, base_dir, items)
            process_page_images(pdf_path, page, page_num, base_dir, items)


    except Exception as e:
        print(f"Error processing PDF file {pdf_path}: {e}")
