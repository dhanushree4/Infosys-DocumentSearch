from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)


def load_documents():
    # Load all PDF files from data folder
    pdf_loader = DirectoryLoader(
        path="data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    # Load all TXT files from data folder
    text_loader = DirectoryLoader(
        path="data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents = []

    # Automatically detect and load files
    documents.extend(pdf_loader.load())
    documents.extend(text_loader.load())

    print(f"Total documents loaded: {len(documents)}")

    return documents

if __name__ == "__main__":
    docs = load_documents()