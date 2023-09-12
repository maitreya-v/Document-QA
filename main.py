import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
import tempfile
import re
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from io import BytesIO, StringIO

# Set your OpenAI API key
openai.api_key = "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"

# Function to clean and format the sentence
def finish_sentence(sentence):
    cleaned_sentence = re.sub(r'/n', ' ', sentence)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    cleaned_sentence = re.sub(r'\.[^.]*$', '.', cleaned_sentence)
    return cleaned_sentence

# Streamlit app
def main():
    st.title("PDF Text Processing App")
    
    # Upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        st.text("Processing the PDF file...")
        pdf_content_bytesio = BytesIO(pdf_file.read())

        # Save the BytesIO content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content_bytesio.getvalue())
            temp_file_path = temp_file.name
        label = st.text_input("Question:", "")

        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai.api_key)
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

        prompt_template = """
        %INSTRUCTIONS
        Use the following pieces of context to answer the question at the end. Answer as if you are speaking to a friend. Your tone should be warmer. Please answer in detail if you can.
        Please finish the sentence properly and make sure to not have any grammar error.

        %TASK
        Please answer in a friendly tone. Answer in detail.

        {context}

        Question: {question}
        Answer in 300 to 500 words:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        docsearch = Chroma.from_documents(texts, embeddings)
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 50}), chain_type_kwargs=chain_type_kwargs)
        result = qa({"query": label})
        ans = finish_sentence(result['result'])

        # Display the result
        st.header("Output:")
        st.text("The processed answer:")
        st.write(ans)
        
if __name__ == "__main__":
    main()
