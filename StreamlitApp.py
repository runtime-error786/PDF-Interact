import streamlit as st
from src.QA.QA import create_retrieval_qa
from src.QA.util import process_file, split_text
from langchain_community.llms import Ollama

def main():
    st.title("Document Question Answering App")
    
    uploaded_file = st.file_uploader("Upload a file", type=["txt"])
    
    if uploaded_file is not None:
        # Process the uploaded file
        data = uploaded_file.read().decode("utf-8")
        chunks = split_text(data)
        
        # Create Retrieval QA system
        llm = Ollama(model="llama3")
        retrieval_qa = create_retrieval_qa(chunks, llm)
        
        st.write("File processed successfully. Ask your questions below:")
        
        question = st.text_input("Ask a question:")
        
        if st.button("Submit"):
            if question:
                answer = retrieval_qa.run(question)
                st.write("Answer:", answer)
            else:
                st.write("Please enter a question.")
                
if __name__ == "__main__":
    main()
