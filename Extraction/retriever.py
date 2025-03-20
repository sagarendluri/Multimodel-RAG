from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

faiss_text_index_path = "/Users/hemasagarendluri1996/john-william-anna-projects/faiss_text_index.faiss"
# faiss_image_index_path = "faiss_image_index.faiss"

text_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
# image_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

loaded_text_store = FAISS.load_local(faiss_text_index_path, text_embeddings, allow_dangerous_deserialization=True)
# loaded_image_store = FAISS.load_local(faiss_image_index_path, image_embeddings, allow_dangerous_deserialization=True)

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=qa_template,
                        input_variables=['context', 'question'])

llm = Ollama(model="llama3.2")

text_retriever = loaded_text_store.as_retriever() # Make sure you have this line
# image_retriever = loaded_image_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=text_retriever,  
    chain_type_kwargs={'prompt': prompt}
)

question = "can you describe the chart and take all the values and compare them in well way?"
result = qa_chain({"query": question})

print("Answer:", result["result"])

# --- Optionally print the source documents ---
# if 'source_documents' in result:
#     print("\nSource Documents:")
#     for doc in result['source_documents']:
#         print(doc.metadata['source'])