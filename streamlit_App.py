import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. *Enter Your API Key*: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

2. *Upload Your Documents*: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. *Ask a Question*: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
api_key = st.text_input("Enter your Google API Key:", type="password", key="INSERT YOUR GOOGLE API KEY HERE")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """You are an expert for answering questions from PDF Files. Follow these steps to analyze the question and provide an accurate answer based on the given context:

    Step 1: Identify Key Concepts
    - List all key concepts mentioned in the question.
    - Separate these concepts into those likely to be in the context and those that might be irrelevant or incorrect.
    

    Step 2: Scan the Context
    - Search the context for information related to each key concept from Step 1.
    - Note which concepts are present and which are absent in the context.
    

    Step 3: Find Nearest Relevant Information
    - For concepts present in the context, identify the most relevant information.
    - For concepts not directly mentioned, find the closest related information in the context.
    

    Step 4: Address Misconceptions
    - Identify any misconceptions in the question based on the context.
    - Prepare to correct these misconceptions using information from the context.
    

    Step 5: Formulate the Answer
    - Start by addressing any misconceptions or incorrect assumptions.
    - Provide information on the concepts present in the context, even if they differ from what was asked.
    - Explain why certain asked concepts are not applicable, if relevant.
    - Use the nearest relevant information for concepts not directly addressed in the context.
    

    Step 6: Summarize and Clarify
    - Provide a clear summary of the correct information from the context.
    - Explicitly state which parts of the question could not be answered from the given context.
   

    Step 7: Generate a final answer 
    -give a final answer based on step 6 and return the answer

    Remember:
    - Do not invent or assume information not provided.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain



def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    print("Context being used for the query:")
    print(context)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
