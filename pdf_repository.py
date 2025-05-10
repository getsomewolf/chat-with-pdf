from langchain_community.document_loaders import PyPDFLoader

class PDFRepository:
    def load(self, pdf_path: str):
        """Carrega o PDF e retorna lista de Document."""
        loader = PyPDFLoader(file_path=pdf_path, extract_images=True)
        documents = loader.load()
        return documents
