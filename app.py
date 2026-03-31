import faiss
import numpy as np

from pypdf import PdfReader
from docx import Document

from sentence_transformers import SentenceTransformer


# ----------------------------
# READ FILE FUNCTIONS
# ----------------------------

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        content = page.extract_text()

        if content:

            text += content

    return text



def read_docx(file_path):

    doc = Document(file_path)

    text = ""

    for para in doc.paragraphs:

        text += para.text + "\n"

    return text



def read_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as f:

        return f.read()



# ----------------------------
# LOAD DOCUMENT
# ----------------------------

def load_document(file_path):

    if file_path.endswith(".pdf"):

        return read_pdf(file_path)

    elif file_path.endswith(".docx"):

        return read_docx(file_path)

    elif file_path.endswith(".txt"):

        return read_txt(file_path)

    else:

        raise ValueError("Unsupported file format")



# ----------------------------
# SPLIT TEXT INTO SENTENCES
# ----------------------------

import re

def split_text(text):

    sentences = re.split(r'[.!?]', text)

    return [s.strip() for s in sentences if len(s.strip()) > 20]



# ----------------------------
# MAIN PROGRAM
# ----------------------------

def main():

    file_path = input("Enter file path: ")

    print("\nReading document...")

    text = load_document(file_path)

    sentences = split_text(text)

    print("Total sentences:", len(sentences))


    print("\nCreating embeddings...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(sentences)


    # create vector database

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))


    print("\nDocument ready! Ask questions.")


    # question loop

    while True:


        query = input("\nQuestion (type exit): ")


        if query.lower() == "exit":

            break


        # convert question to embedding

        query_embedding = embedding_model.encode([query])


        # search most relevant sentence

        distances, indices = index.search(query_embedding, 3)

        print("\nBest answers:\n")

        for i in indices[0]:
            print("-", sentences[i])
        print("\n--------------------")



# ----------------------------

if __name__ == "__main__":

    main()
