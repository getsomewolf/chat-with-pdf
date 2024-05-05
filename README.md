# chat-with-pdf

Chat with pdf using Local VectorStore (FAISS)

# Document Retrieval and Question Answering with PDF

This Python script demonstrates document retrieval and question answering capabilities using PDF documents.

FAISS (Facebook AI Similarity Search) is an open-source library developed by Facebook AI Research, designed to efficiently search for similarities in large datasets of high-dimensional vectors.
It specializes in clustering large amounts of data and quickly retrieving items similar to a query. FAISS uses optimized algorithms to accelerate the search process and reduce memory usage.

## Steps:

1. **Environment Setup**: The script loads environment variables from a `.env` file using `dotenv`.

2. **Document Loading**: It uses `PyPDFLoader` from `langchain` to load a PDF document named "react.pdf".

3. **Text Splitting**: The loaded document is split into smaller chunks of text using `CharacterTextSplitter` from `langchain`. Each chunk has a size of 1000 characters with a 30-character overlap and is separated by newline characters.

4. **Embeddings Generation**: OpenAI embeddings are generated for the text chunks using `OpenAIEmbeddings` from `langchain`.

5. **Vector Store Creation**: A vector store is created using `FAISS` from `langchain_community.vectorstores`. The vector store is populated with the text embeddings generated in the previous step.

6. **Saving Vector Store**: The created vector store is saved locally with the name "faiss_index_react".

7. **Loading Vector Store**: The saved vector store is loaded back into memory using `FAISS` with dangerous deserialization enabled.

8. **Question Answering Setup**: A retrieval-based question answering model is initialized using `RetrievalQA` from `langchain.chains`. The model is configured with an OpenAI language model (LLM) and the loaded vector store as a retriever.

9. **Question Answering**: A sample question ("What are the disadvantages of using React?") is passed to the question answering model for inference.

10. **Print Result**: The result of the question answering process is printed to the console.

## Usage:
- Ensure that the "react.pdf" file exists in the current directory.
- Install the required dependencies specified in the script (e.g., `langchain`, `dotenv`, etc.).
- Run the script to perform document retrieval and question answering on the provided PDF document.
