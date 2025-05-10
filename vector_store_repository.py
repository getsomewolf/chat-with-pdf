import os
from langchain_community.vectorstores import FAISS

class VectorStoreRepository:
    def __init__(self, storage_path: str, embeddings):
        self.storage_path = storage_path
        self.embeddings = embeddings
        self.index = None

    def exists(self) -> bool:
        return os.path.exists(self.storage_path) and bool(os.listdir(self.storage_path))

    def load(self):
        """Carrega índice FAISS existente"""
        self.index = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
        return self.index

    def save(self, index):
        """Salva índice FAISS no storage"""
        os.makedirs(self.storage_path, exist_ok=True)
        index.save_local(self.storage_path)

    def create(self, documents: list):
        """Cria um novo índice FAISS a partir de documentos"""
        index = FAISS.from_documents(documents, self.embeddings)
        self.save(index)
        self.index = index
        return index