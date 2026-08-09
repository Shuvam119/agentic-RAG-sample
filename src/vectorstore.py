from langchain_chroma import Chroma

from src.embeddings import get_embeddings


CHROMA_DIR = "chroma_db"


def create_vectorstore(documents):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


def get_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
