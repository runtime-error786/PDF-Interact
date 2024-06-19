from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_file(file):
    return file.read().decode("utf-8")

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    return text_splitter.split_text(data)
