import os
import re
import shutil
import time
import asyncio
import cachetools
from typing import Optional, List, Tuple, AsyncGenerator
import pickle

from langchain_core.documents import Document

from src.config.settings import settings
from src.infra.pdf_repository import PDFRepository
from src.infra.embeddings_factory import EmbeddingFactory
from src.infra.chunk_strategies import ChunkStrategyFactory, ChunkStrategy
from src.infra.vector_store_repository import VectorStoreRepository
from src.infra.retriever_strategies import HybridRetrieverStrategy, RetrieverStrategy # Add others if needed
from src.core.prompt_builder import PromptBuilder
from src.core.llm_client import LLMClient
from src.core.event_manager import EventManager
from src.utils.format_utils import format_docs_for_api, format_docs_for_cli
from src.utils.cli_utils import LoadingIndicator # For CLI usage, might be conditional
from src.utils.file_utils import ensure_pdf_is_in_pdfs_dir
import logging

logger = logging.getLogger(__name__)

class IndexService:
    def __init__(
        self,
        pdf_path: str, # Original path or filename
        event_manager: EventManager,
        force_reindex: bool = False
    ):
        self.original_pdf_path = pdf_path
        self.pdf_path_in_managed_dir = ensure_pdf_is_in_pdfs_dir(pdf_path)
        
        self.pdf_basename = os.path.basename(self.pdf_path_in_managed_dir).split('.')[0]
        self.index_path = os.path.join(settings.INDICES_DIR, f"index_{self.pdf_basename}")
        
        self.event_manager = event_manager
        self.force_reindex = force_reindex

        self.pdf_repo = PDFRepository()
        self.embedding_model = EmbeddingFactory.get_model(
            settings.EMBEDDING_MODEL_NAME, 
            show_progress=False # Typically false for server-side
        )
        self.vs_repo = VectorStoreRepository(
            storage_path=self.index_path, 
            embeddings=self.embedding_model
        )
        self.chunk_strategy: ChunkStrategy = ChunkStrategyFactory.get_strategy(
            mode=settings.CHUNKING_MODE,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        self.vector_store = None
        self.all_chunks: Optional[List[Document]] = None
        logger.info(f"IndexService initialized for PDF: {self.pdf_path_in_managed_dir}, Force reindex: {self.force_reindex}")

    async def initialize_index(self, use_cli_indicator: bool = False):
        """Loads an existing index or creates a new one if needed or forced."""
        self.event_manager.emit('index_setup_started', {'pdf_path': self.pdf_path_in_managed_dir, 'force_reindex': self.force_reindex})

        if self.vs_repo.exists() and not self.force_reindex:
            logger.info(f"Index found for {self.pdf_path_in_managed_dir}. Loading...")
            indicator_msg = "Carregando índice existente..."
            if use_cli_indicator:
                with LoadingIndicator(indicator_msg):
                    self.vector_store, self.all_chunks = await asyncio.to_thread(self.vs_repo.load)
            else:
                self.vector_store, self.all_chunks = await asyncio.to_thread(self.vs_repo.load)

            logger.info(f"Index loaded successfully from {self.index_path}")
            self.event_manager.emit('index_loaded', {'path': self.index_path})
        else:
            if self.force_reindex and self.vs_repo.exists():
                logger.info(f"Forcing re-creation of index for {self.pdf_path_in_managed_dir}.")
            else:
                logger.info(f"No index found or re-index forced for {self.pdf_path_in_managed_dir}. Creating new index.")
            await self._create_index(use_cli_indicator)

        self.event_manager.emit('index_setup_completed', {'pdf_path': self.pdf_path_in_managed_dir})

    async def _ensure_chunks_available_for_retriever(self, use_cli_indicator: bool = False):
        """Ensures self.all_chunks is populated, loading from disk if available."""
        chunks_path = os.path.join(self.index_path, "index_chunks.pkl")
        if self.all_chunks is None:
            if os.path.exists(chunks_path):
                logger.info("Loading chunks from disk.")
                with open(chunks_path, "rb") as f:
                    self.all_chunks = pickle.load(f)
            else:
                logger.info("Chunks not found on disk. Processing PDF to generate chunks.")
                indicator_msg = "Lendo PDF para chunks..."
                if use_cli_indicator:
                    with LoadingIndicator(indicator_msg):
                        documents = await asyncio.to_thread(self.pdf_repo.load, self.pdf_path_in_managed_dir)
                else:
                    documents = await asyncio.to_thread(self.pdf_repo.load, self.pdf_path_in_managed_dir)

                self.all_chunks = await asyncio.to_thread(self.chunk_strategy.split, documents)
                for i, doc_chunk in enumerate(self.all_chunks):
                    doc_chunk.metadata['chunk_index'] = i + 1

                # Save chunks to disk for future use
                with open(chunks_path, "wb") as f:
                    pickle.dump(self.all_chunks, f)

                logger.info(f"PDF processed into {len(self.all_chunks)} chunks for retriever.")


    async def _create_index(self, use_cli_indicator: bool = False):
        logger.info(f"Starting PDF processing for indexing: {self.pdf_path_in_managed_dir}")
        self.event_manager.emit('index_creation_started', {'path': self.index_path})

        indicator_msg_pdf = "Lendo PDF..."
        if use_cli_indicator:
            with LoadingIndicator(indicator_msg_pdf):
                documents = await asyncio.to_thread(self.pdf_repo.load, self.pdf_path_in_managed_dir)
        else:
            documents = await asyncio.to_thread(self.pdf_repo.load, self.pdf_path_in_managed_dir)
        
        logger.info(f"PDF loaded: {len(documents)} pages.")
        # Add document stats logging here if desired

        self.all_chunks = await asyncio.to_thread(self.chunk_strategy.split, documents)
        for i, doc_chunk in enumerate(self.all_chunks): # Add chunk_index metadata
            doc_chunk.metadata['chunk_index'] = i + 1

        logger.info(f"Document split into {len(self.all_chunks)} chunks using mode: {settings.CHUNKING_MODE}")
        self.event_manager.emit('chunks_split', {'count': len(self.all_chunks), 'mode': settings.CHUNKING_MODE})

        indicator_msg_faiss = "Criando vetores e índice FAISS..."
        if use_cli_indicator:
            with LoadingIndicator(indicator_msg_faiss):
                self.vector_store = await asyncio.to_thread(self.vs_repo.create, self.all_chunks)
        else:
            self.vector_store = await asyncio.to_thread(self.vs_repo.create, self.all_chunks)
            
        logger.info(f"FAISS index created and saved to {self.index_path}")
        self.event_manager.emit('index_created', {'path': self.index_path})

    def get_vector_store(self):
        if not self.vector_store:
            # This should ideally not happen if initialize_index was called.
            logger.warning("Accessing vector_store before initialization. Attempting to load.")
            # Fallback to synchronous load for simplicity here, but async path is preferred.
            if self.vs_repo.exists():
                self.vector_store = self.vs_repo.load()
            else:
                raise RuntimeError("Vector store not initialized and no existing index found.")
        return self.vector_store

    def get_all_chunks(self) -> Optional[List[Document]]:
        if self.all_chunks is None:
            # This implies initialize_index or _ensure_chunks_available_for_retriever wasn't fully effective
            logger.warning("Accessing all_chunks before they are populated.")
            # Consider a synchronous load or raise error
            # For now, returning None, but this needs robust handling
            return None
        return self.all_chunks


class QueryService:
    def __init__(
        self,
        vector_store, # Should be a fully initialized FAISS index
        all_chunks: List[Document], # Needed for BM25 part of HybridRetriever
        event_manager: EventManager,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient
    ):
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self.event_manager = event_manager
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

        self.response_cache = cachetools.TTLCache(
            maxsize=settings.RESPONSE_CACHE_MAX_SIZE,
            ttl=settings.RESPONSE_CACHE_TTL_SECONDS
        )
        
        # Setup Retriever Strategy
        if not self.all_chunks:
            logger.error("Cannot initialize HybridRetrieverStrategy: all_chunks is None or empty.")
            # Fallback or raise error. For now, we'll let it potentially fail if used.
            # This indicates an issue in IndexService's chunk provisioning.
            self.retriever_strategy: Optional[RetrieverStrategy] = None
        else:
            self.retriever_strategy: RetrieverStrategy = HybridRetrieverStrategy(
                vector_store=self.vector_store,
                threshold=settings.VECTOR_DISTANCE_THRESHOLD,
                initial_k=settings.INITIAL_VECTOR_K,
                final_k=settings.FINAL_BM25_K,
                documents=self.all_chunks
            )
        logger.info("QueryService initialized.")

    def _decompose_complex_query(self, query: str) -> List[str]:
        """Decomposes a complex query into simpler sub-queries."""
        # Basic decomposition, can be expanded
        if '?' in query and query.count('?') > 1:
            return [q.strip() + '?' for q in query.split('?') if q.strip()]
        
        conjunctions = [' and ', ' or ', ' e ', ' ou ', ', '] # Added comma
        for conj in conjunctions:
            if conj in query.lower():
                # More robust splitting needed if conjunctions are part of phrases
                return [q.strip() for q in re.split(f'{conj}', query, flags=re.IGNORECASE) if q.strip()]
        return [query]

    async def _retrieve_documents(self, question: str) -> List[Document]:
        if not self.retriever_strategy:
            logger.error("Retriever strategy not available.")
            return []
            
        self.event_manager.emit('retrieval_started', {'question': question})
        start_time = time.time()

        # Decompose query if complex
        sub_queries = self._decompose_complex_query(question)
        
        all_retrieved_docs: List[Document] = []
        if len(sub_queries) > 1:
            logger.info(f"Decomposed query into {len(sub_queries)} parts: {sub_queries}")
            for sub_q in sub_queries:
                # Run retrieval in thread pool as it might have sync parts
                docs = await asyncio.to_thread(self.retriever_strategy.retrieve, sub_q)
                all_retrieved_docs.extend(docs)
            # Deduplicate documents
            seen_content_hashes = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content_hashes:
                    unique_docs.append(doc)
                    seen_content_hashes.add(content_hash)
            final_docs = unique_docs
        else:
            final_docs = await asyncio.to_thread(self.retriever_strategy.retrieve, question)

        retrieval_time = time.time() - start_time
        logger.info(f"Retrieval completed in {retrieval_time:.2f}s, found {len(final_docs)} documents.")
        self.event_manager.emit('retrieval_completed', {'count': len(final_docs), 'time': retrieval_time})
        return final_docs

    async def answer_question_streaming(self, question: str) -> AsyncGenerator[Tuple[str, Optional[dict]], None]:
        """
        Answers a question by retrieving documents, then generating a response with LLM, streaming results.
        Yields tuples of (event_type, data_dict).
        Example events: ("text_chunk", {"chunk": "..."}), ("sources", {"sources": [...]})
        """
        cached_response = self.response_cache.get(question)
        if cached_response and isinstance(cached_response, dict) and 'answer_stream_complete' in cached_response:
            logger.info(f"Full answer stream for '{question}' found in cache.")
            # This simplistic cache re-yields stored chunks. A real SSE cache might be more complex.
            for part in cached_response.get('streamed_parts', []):
                yield part # part is (event_type, data_dict)
            return

        logger.info(f"Processing question (streaming): '{question}'")
        
        final_docs = await self._retrieve_documents(question)
        if not final_docs:
            logger.warning("No relevant documents found for the question.")
            yield ("text_chunk", {"chunk": "Não foram encontrados documentos relevantes para essa pergunta."})
            yield ("sources", {"sources": []})
            return

        context_text, sources_list = format_docs_for_api(final_docs)
        # Yield sources first
        yield ("sources", {"sources": sources_list})
        
        # Store parts for potential caching
        streamed_parts_for_cache = [("sources", {"sources": sources_list})]

        if not context_text.strip():
            logger.warning("Context text is empty after formatting documents.")
            yield ("text_chunk", {"chunk": "O contexto extraído dos documentos está vazio."})
            return

        logger.info(f"Context generated: {len(context_text)} chars. Asking LLM.")
        
        full_answer = ""
        async for answer_chunk in self.llm_client.generate(context_text, question):
            yield ("text_chunk", {"chunk": answer_chunk})
            streamed_parts_for_cache.append(("text_chunk", {"chunk": answer_chunk}))
            full_answer += answer_chunk
        
        # Cache the full streamed response if needed (more complex for generators)
        # For simplicity, we'll cache the final answer and sources for non-streaming,
        # and a marker that the stream was completed.
        self.response_cache[question] = {
            'answer_stream_complete': True, 
            'final_answer_for_reference': full_answer, # Not directly used by streaming client
            'sources': sources_list,
            'streamed_parts': streamed_parts_for_cache # For re-yielding if cached
        }

    async def answer_question_non_streaming(self, question: str, use_cli_formatting: bool = False) -> Tuple[str, List[str]]:
        """Answers a question, returns full answer and sources (non-streaming)."""
        cached_response = self.response_cache.get(question)
        if cached_response and isinstance(cached_response, dict) and 'final_answer' in cached_response:
            logger.info(f"Answer for '{question}' found in cache.")
            return cached_response['final_answer'], cached_response['sources']

        logger.info(f"Processing question (non-streaming): '{question}'")
        
        final_docs = await self._retrieve_documents(question)
        if not final_docs:
            logger.warning("No relevant documents found.")
            return "Não foram encontrados documentos relevantes para essa pergunta.", []

        if use_cli_formatting:
            # format_docs_for_cli prints, so we need a version that returns text for context
            # For now, let's use a simplified context for CLI or adapt format_docs_for_cli
            context_text_cli = "\n\n".join([doc.page_content for doc in final_docs])
            # Call the printing version for CLI feedback
            format_docs_for_cli(final_docs) # This prints to console
            context_text = context_text_cli
            _, sources_list = format_docs_for_api(final_docs) # Get sources consistently
        else:
            context_text, sources_list = format_docs_for_api(final_docs)

        if not context_text.strip():
            logger.warning("Context text is empty after formatting documents.")
            return "O contexto extraído dos documentos está vazio.", []
        
        logger.info(f"Context generated: {len(context_text)} chars. Asking LLM (non-streaming).")
        
        answer = await self.llm_client.generate_non_streaming(context_text, question)
        
        self.response_cache[question] = {'final_answer': answer, 'sources': sources_list}
        return answer, sources_list
