import os
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings  # Example embedding model
from langchain_community.vectorstores import FAISS

data_path = "media"

# --- Configuration ---
faiss_text_index_path = "faiss_text_index.faiss"
# faiss_image_index_path = "faiss_image_index.faiss"


# qdrant_client.QdrantClient(path=qdrant_path)
# client = qdrant_client.QdrantClient(path=qdrant_path)


text_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
image_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

loader = DirectoryLoader(data_path)
documents = loader.load()
# --- Load Documents Manually (to ensure file closure) ---
print("document ", documents)
all_documents = documents
for root, _, files in os.walk(data_path):
    for file in files:
        file_path = os.path.join(root, file)
        print("file_paths",file_path)
        metadata = {"source": file_path}
        if file_path.endswith(('.txt')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_documents.append(Document(page_content=content, metadata=metadata))
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r') as f: # Try without explicit encoding
                        content = f.read()
                        all_documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    print(f"Error reading text file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            # For image files, we'll create a Document with empty content for now.
            # The focus here is to avoid the decoding error.
            all_documents.append(Document(page_content="", metadata=metadata))

# --- Separate Documents ---
text_documents = [
    doc for doc in all_documents
    if "media/text/" in doc.metadata.get('source', '') and doc.metadata.get('source', '').endswith(('.txt'))
    or "media/tables/" in doc.metadata.get('source', '') and doc.metadata.get('source', '').endswith(('.txt'))
]
# image_documents = [
#     doc for doc in all_documents
#     if "media/images/" in doc.metadata.get('source', '') and doc.metadata.get('source', '').endswith(('.png', '.jpg', '.jpeg'))
# ]

print(f"Number of text documents found: {text_documents}")
# print(f"Number of image documents found: {image_documents}")

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

print("all_documents", all_documents)
# loader = TextLoader("media/text/advanced_2_pdf_text_0_0.txt")
# documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0)
docs = text_splitter.split_documents(all_documents)
print(docs)
# For Text
text_store = FAISS.from_documents(
    text_documents,
    text_embeddings,
)
# # For Images
# image_store = FAISS.from_documents(
#     image_documents,
#     image_embeddings,
# )
print(f"Image documents loaded and indexed into FAISS.")

# --- Optional: Save the FAISS index to disk ---
FAISS.save_local(text_store, faiss_text_index_path)
# FAISS.save_local(image_store, faiss_image_index_path)
print(f"FAISS index for text saved to: {faiss_text_index_path}")
# print(f"FAISS index for images saved to: {faiss_image_index_path}")
