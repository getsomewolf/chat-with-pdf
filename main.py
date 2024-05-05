from dotenv import load_dotenv

load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    print("Chat with PDF")
    pdf_path = "react.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("What are the disadvantages of using React?")
    print(res)
