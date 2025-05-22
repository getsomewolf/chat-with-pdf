from abc import ABC, abstractmethod
import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkStrategy(ABC):
    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        pass

class TokenChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

class ParagraphChunkStrategy(ChunkStrategy):
    def split(self, documents: list[Document]) -> list[Document]:
        paragraph_docs = []
        for doc in documents:
            page = doc.metadata.get('page')
            paras = re.split(r"\n\n+|(?<=\.)\s*\n(?=[A-ZÁÉÍÓÚÃÕÂÊÎÔÛ])", doc.page_content)
            for idx, para in enumerate(paras):
                text = para.strip()
                if text:
                    meta = dict(doc.metadata)
                    meta['page'] = page
                    meta['paragraph'] = idx + 1
                    paragraph_docs.append(Document(page_content=text, metadata=meta))
        return paragraph_docs

class CombinedChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.paragraph = ParagraphChunkStrategy()
        self.token = TokenChunkStrategy(chunk_size, chunk_overlap)
    def split(self, documents: list[Document]) -> list[Document]:
        paras = self.paragraph.split(documents)
        return self.token.split(paras)

class ChunkStrategyFactory:
    @staticmethod
    def get_strategy(mode: str, chunk_size: int = None, chunk_overlap: int = None) -> ChunkStrategy:
        if mode == 'tokens':
            return TokenChunkStrategy(chunk_size, chunk_overlap)
        if mode == 'paragraphs':
            return ParagraphChunkStrategy()
        return CombinedChunkStrategy(chunk_size, chunk_overlap)
