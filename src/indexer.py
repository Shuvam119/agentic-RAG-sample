from src.ingestion import load_documents, split_documents
from src.vectorstore import create_vectorstore


def build_index():

    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents/pages.")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating vector database...")
    create_vectorstore(chunks)

    print("Index created successfully.")


if __name__ == "__main__":
    build_index()
