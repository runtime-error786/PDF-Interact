from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as P1
from pinecone import Pinecone as p1
from langchain_pinecone import PineconeVectorStore
import os

def create_retrieval_qa(chunks, llm):
    os.environ['PINECONE_API_KEY'] = '39c3b55b-2ae4-44ee-a9cd-83a99876c828'
    pc = p1(api_key=os.environ.get("PINECONE_API_KEY"))

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = "test1"

    index = pc.Index(index_name)
    for i, t in zip(range(len(chunks)), chunks):
        query_result = embedding.embed_query(t)
        index.upsert(
            vectors=[
                {
                    "id": str(i),
                    "values": query_result,
                    "metadata": {"text": str(t)}
                }
            ],
            namespace="real"
        )
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding, namespace="real")
    retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    
    return retrieval_qa
