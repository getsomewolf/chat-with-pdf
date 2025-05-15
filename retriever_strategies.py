from abc import ABC, abstractmethod
from langchain_community.retrievers import BM25Retriever

class RetrieverStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list:
        pass

class VectorRetrieverStrategy(RetrieverStrategy):
    def __init__(self, vector_store, k: int):
        self.retriever = vector_store.as_retriever(search_kwargs={'k': k})
    def retrieve(self, query: str) -> list:
        return self.retriever.invoke(query)

class BM25RetrieverStrategy(RetrieverStrategy):
    def __init__(self, documents: list, k: int):
        self.retriever = BM25Retriever.from_documents(documents, k=k)
    def retrieve(self, query: str) -> list:
        return self.retriever.invoke(query)

class HybridRetrieverStrategy(RetrieverStrategy):
    def __init__(self, vector_store, threshold: float, initial_k: int, final_k: int, documents: list):
        self.vector_store = vector_store
        self.threshold = threshold
        self.initial_k = initial_k
        self.final_k = final_k
        self.documents = documents

    def retrieve(self, query: str) -> list:
        # passo 1: busca vetorial ampla com scores
        initial = self.vector_store.similarity_search_with_score(query, k=self.initial_k)
        # passo 2: filtrar por threshold
        filtered = [doc for doc, score in initial if score < self.threshold]
        if not filtered:
            # fallback top final_k vetoriais
            return [doc for doc, _ in initial][:self.final_k]
        # passo 3: BM25 sobre filtrados
        bm25 = BM25Retriever.from_documents(filtered, k=min(self.final_k, len(filtered)))
        return bm25.invoke(query)