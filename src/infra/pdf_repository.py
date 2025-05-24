from langchain_community.document_loaders import PyPDFLoader
import logging
# Import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader


logger = logging.getLogger(__name__)

class PDFRepository:
    def load(self, pdf_path: str):
        """Carrega o PDF e retorna lista de Document."""
        # Use PyMuPDFLoader instead of PyPDFLoader
        loader = PyMuPDFLoader(file_path=pdf_path)
        try:
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents from {pdf_path} using PyMuPDFLoader.")
            if not documents:
                logger.warning(f"No documents were loaded from {pdf_path}. The PDF might be empty or unparseable.")
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path} with PyMuPDFLoader: {e}", exc_info=True)
            # Retornar uma lista vazia ou levantar a exceção, dependendo de como você quer lidar com PDFs problemáticos
            # Para este caso, vamos retornar uma lista vazia para que o IndexService possa lidar com isso.
            return []
        return documents
