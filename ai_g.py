import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from PyPDF2 import PdfReader
# from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader, AsyncHtmlLoader, PyPDFLoader, Docx2txtLoader, \
    WebBaseLoader, AsyncChromiumLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
st.set_page_config(page_title="AI RAG", layout="wide")

st.title("Hai Welcometo AI RAG")

st.sidebar.title("Train Your Data")
st.sidebar.markdown("Choose the content and upload the data")


def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not available "i donot know what the hell you are asking about"
         """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


# google_api_key = st.sidebar.text_input("Google API Key", placeholder="Enter Google API Key")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # new_db = FAISS.load_local("faiss_indexl", embeddings, allow_dangerous_deserialization=True)
    # docs = new_db.similarity_search(user_question)
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # chain = get_conversational_chain()
    # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # st.write("Reply: ", response["output_text"])
    # response = model.invoke(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)
    # retrieval_chain = RetrievalQA.from_chain_type(
    #     llm, chain_type="stuff", retriever=vectordb.as_retriever(), return_source_documents=True, verbose=False
    # )
    # result = retrieval_chain.invoke(user_question)
    prompt = load_prompt()

    rag_chain = (
            {"context": vectordb.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    response = rag_chain.invoke(user_question)
    print("response ghjgjh",response)

    st.write("Reply: ", response)


def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    if user_question:  # Ensure API key and user question are provided
        user_input(user_question)
    else:
        st.warning("Please provide a question and API key.", icon="⚠️")

    with st.sidebar:
        st.title("Train ur Data:")

        # Additional Text Input Field
        additional_text = st.sidebar.text_input("Plain Text / Webpage URL",
                                                placeholder="Enter Plain Text / Webpage URL")

        # Select Box for File Type
        file_type = st.sidebar.selectbox(
            "Select File Type",
            ("Plain Text File", "Webpage", "File Upload")
        )

        # File Uploader
        uploaded_file = st.sidebar.file_uploader("Upload a File")
        if st.button("Submit & Process",
                     key="process_button"):  # Check if API key is provided before processing
            # if not google_api_key:
            #     st.warning("Please provide a Google API Key.",icon="⚠️")
            # else:
            with st.spinner("Processing..."):
                # raw_text = get_pdf_text(pdf_docs)
                # text_chunks = get_text_chunks(raw_text)
                # get_vector_store(text_chunks, api_key)

                import tempfile


                if file_type == "Plain Text File":
                    if not additional_text:
                        st.warning("Please provide a Plain Text File.", icon="⚠️")
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                        temp_file.write(additional_text)
                        temp_file_path = temp_file.name
                    with open(temp_file_path) as file:
                        content = file.read()
                    loader = TextLoader(temp_file_path)
                    documents = loader.load()

                elif file_type == "Webpage":
                    if not additional_text:
                        st.warning("Please provide a Webpage URL.", icon="⚠️")

                    loader = AsyncHtmlLoader(["https://react.dev/learn/installation"])
                    # loader = AsyncHtmlLoader(["https://react.dev/learn/installation"])
                    documents = loader.load()
                elif file_type == "File Upload":
                    print("uploaded_fileuploaded_file",uploaded_file)
                    print("uploaded_fileuploaded_file",uploaded_file.name)
                    if not uploaded_file:
                        st.warning("Please provide a File.", icon="⚠️")
                    if uploaded_file.name.endswith(".pdf"):
                        temp_file = "./temp.pdf"
                        with open(temp_file, "wb") as file:
                            file.write(uploaded_file.getvalue())
                            file_name = uploaded_file.name
                        loader = PyPDFLoader(temp_file)
                    elif uploaded_file.name.endswith(".doc") or uploaded_file.name.endswith(".docx"):
                        temp_file = "./temp.doc"
                        with open(temp_file, "wb") as file:
                            file.write(uploaded_file.getvalue())
                            file_name = uploaded_file.name
                        loader = Docx2txtLoader(temp_file)
                    elif uploaded_file.name.endswith(".txt"):
                        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                            text_content = uploaded_file.read().decode("utf-8")
                            temp_file.write(text_content)
                            temp_file_path = temp_file.name
                        with open(temp_file_path) as file:
                            content = file.read()
                        loader = TextLoader(temp_file_path)
                    else:
                        st.warning("Please provide a File Type.", icon="⚠️")
                        return None
                    documents = loader.load()

                else:
                    st.warning("Please provide a File Type.", icon="⚠️")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                vectordb = Chroma.from_documents(documents=docs, embedding=embeddings,
                                                 persist_directory="chroma_db")
                # vectordb.persist()

                st.success("Done")


if __name__ == "__main__":
    main()
