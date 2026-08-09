from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_metadata(file_path: Path):
    name = file_path.stem

    metadata = {
        "source": file_path.name,
        "product": "CloudDesk",
    }

    if "v1.0" in name:
        metadata["version"] = "1.0"

    elif "v2.0" in name:
        metadata["version"] = "2.0"

    elif "v2.2" in name:
        metadata["version"] = "2.2"

    if "API_Reference" in name:
        metadata["document_type"] = "API Reference"

    elif "User_Guide" in name:
        metadata["document_type"] = "User Guide"

    elif "Administrator" in name:
        metadata["document_type"] = "Administrator Guide"

    elif "Troubleshooting" in name:
        metadata["document_type"] = "Troubleshooting"

    elif "Installation" in name:
        metadata["document_type"] = "Installation Guide"

    elif "Release_Notes" in name:
        metadata["document_type"] = "Release Notes"

    elif "Known_Issues" in name:
        metadata["document_type"] = "Known Issues"

    if "v1.0" in name:
        metadata["status"] = "Archived"
    else:
        metadata["status"] = "Current"

    if "User_Guide" in name:
        metadata["audience"] = "End Users"

    elif "Administrator" in name:
        metadata["audience"] = "Administrators"

    elif "Installation" in name:
        metadata["audience"] = "Administrators"

    elif "API_Reference" in name:
        metadata["audience"] = "Developers"

    elif "Troubleshooting" in name:
        metadata["audience"] = "Support and Administrators"

    elif "Release_Notes" in name:
        metadata["audience"] = "All Audiences"

    elif "Known_Issues" in name:
        metadata["audience"] = "Support and Administrators"

    return metadata


def load_documents(data_dir="data/raw"):
    documents = []

    for file_path in Path(data_dir).iterdir():

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif file_path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file_path))

        else:
            continue

        loaded_documents = loader.load()

        metadata = extract_metadata(file_path)

        for document in loaded_documents:
            document.metadata.update(metadata)

        documents.extend(loaded_documents)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.split_documents(documents)
