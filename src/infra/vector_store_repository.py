import os
import pickle
from langchain_community.vectorstores import FAISS

class VectorStoreRepository:
    def __init__(self, storage_path: str, embeddings):
        self.storage_path = storage_path
        self.embeddings = embeddings
        self.index = None

    def exists(self) -> bool:
        return os.path.exists(self.storage_path) and bool(os.listdir(self.storage_path))

    def load(self):
        """Carrega índice FAISS existente e chunks associados."""
        self.index = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
        chunks_path = os.path.join(self.storage_path, "index_chunks.pkl")
        chunks = None
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
        return self.index, chunks

    def save(self, index, chunks):
        """Salva índice FAISS e chunks associados no storage."""
        os.makedirs(self.storage_path, exist_ok=True)
        index.save_local(self.storage_path)
        chunks_path = os.path.join(self.storage_path, "index_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

    def create(self, documents: list):
        """Cria um novo índice FAISS a partir de documentos e salva os chunks."""
        index = FAISS.from_documents(documents, self.embeddings)
        self.save(index, documents)
        self.index = index
        return index