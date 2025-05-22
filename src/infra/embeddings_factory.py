from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings

class EmbeddingFactory:
    @staticmethod
    def get_model(name: str, show_progress: bool = True):
        """Retorna instância de embeddings"""
        return HFEmbeddings(model_name=name, show_progress=show_progress)