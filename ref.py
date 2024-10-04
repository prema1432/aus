import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = 'llm_db/db_faiss'

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Open API key for the chatbot to accessChat Gpt Models. 

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

api_key = st.text_input("Enter your Open API Key:", type="password", key="api_key_input")


def load_knowledgeBase():
    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)
    return llm


# creating prompt template using langchain
def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf answer "i donot know what the hell you are asking about"
         """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def user_input(user_question):
    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()
    similar_embeddings = knowledgeBase.similarity_search(user_question)
    similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
                                              embedding=OpenAIEmbeddings(
                                                  api_key=api_key))

    # creating the chain for integrating llm,prompt,stroutputparser
    retriever = similar_embeddings.as_retriever()
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    response = rag_chain.invoke(user_question)
    st.write(response)


def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    key="pdf_uploader", type="pdf")
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                temp_file = "./temp.pdf"
                with open(temp_file, "wb") as file:
                    file.write(pdf_docs.getvalue())
                    file_name = pdf_docs.name

                loader = PyPDFLoader(temp_file)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(
                    openai_api_key=api_key))
                vectorstore.save_local(DB_FAISS_PATH)

                st.success("Done")


if __name__ == "__main__":
    main()