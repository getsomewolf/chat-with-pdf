# filepath: c:\Users\lucas\Projects\chat-with-pdf\api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import asyncio
import json
from typing import Dict, Tuple, AsyncGenerator

from src.config.settings import settings
from src.core.services import IndexService, QueryService
from src.core.event_manager import EventManager
from src.core.observers import LoggingObserver
from src.core.prompt_builder import PromptBuilder
from src.core.llm_client import LLMClient

import logging
import uvicorn
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chat with PDF API",
    version="1.1.0",
    description="API for uploading PDFs and asking questions about their content."
)

# Global cache for service instances, keyed by PDF filename (basename)
# This is a simple in-memory cache. For production, consider a more robust solution
# or manage service lifecycles with FastAPI dependencies if appropriate.
service_instances_cache: Dict[str, Tuple[IndexService, QueryService]] = {}

# Setup global event manager and logger for API context
api_event_manager = EventManager()
api_logger = LoggingObserver() # Using the general LoggingObserver
event_types_to_log = [
    'index_setup_started', 'index_loaded', 'index_creation_started', 'chunks_split', 
    'index_created', 'index_setup_completed', 'retrieval_started', 'retrieval_completed', 
    'generation_started', 'generation_completed', 'generation_failed', 'api_upload_request',
    'api_ask_request'
]
for ev_type in event_types_to_log:
    api_event_manager.subscribe(ev_type, api_logger)

# --- Pydantic Models ---
class UploadResponse(BaseModel):
    message: str
    pdf_filename: str
    index_status: str

class QuestionRequest(BaseModel):
    pdf_filename: str # Basename of the PDF, e.g., "mydoc.pdf"
    question: str

# SSE Stream Event Model (conceptual, not directly used by Pydantic for StreamingResponse)
# class SSEEvent(BaseModel):
#     event: str
#     data: str # JSON stringified data

# --- Helper Function to Get or Create Services ---
async def get_or_create_services(pdf_filename_basename: str, force_reindex: bool = False) -> Tuple[IndexService, QueryService]:
    if pdf_filename_basename in service_instances_cache and not force_reindex:
        logger.info(f"Using cached services for {pdf_filename_basename}")
        # Potentially re-validate if index still exists or needs refresh if not forcing
        # For simplicity, we return cached if not forcing reindex.
        # A more robust check might involve IndexService.is_valid() or similar.
        return service_instances_cache[pdf_filename_basename]

    # Path to the PDF within the managed PDFS_DIR
    pdf_path_in_managed_dir = os.path.join(settings.PDFS_DIR, pdf_filename_basename)
    if not os.path.exists(pdf_path_in_managed_dir) and not force_reindex: # if force_reindex, upload will place it
        # This case happens if /ask is called for a PDF that was never uploaded or its file is gone
        raise FileNotFoundError(f"PDF file {pdf_filename_basename} not found in managed directory: {settings.PDFS_DIR}")


    # Create new instances
    logger.info(f"Creating new service instances for {pdf_filename_basename}. Force reindex: {force_reindex}")
    
    # IndexService needs the original path or the path it will be copied to.
    # If force_reindex is true, it's typically an upload, so pdf_path_in_managed_dir is the target.
    # If not force_reindex, it's a query, so pdf_path_in_managed_dir must exist.
    index_service = IndexService(
        pdf_path=pdf_path_in_managed_dir, # IndexService handles ensuring it's in PDFS_DIR
        event_manager=api_event_manager,
        force_reindex=force_reindex 
    )
    await index_service.initialize_index() # This loads or creates the index

    vector_store = index_service.get_vector_store()
    all_chunks = index_service.get_all_chunks()

    if not vector_store or not all_chunks:
        # This indicates a problem during index initialization
        logger.error(f"Failed to obtain vector_store or all_chunks for {pdf_filename_basename} after initialization.")
        raise HTTPException(status_code=500, detail=f"Index initialization failed for {pdf_filename_basename}. Cannot create QueryService.")

    prompt_builder = PromptBuilder()
    llm_client = LLMClient(event_manager=api_event_manager, prompt_builder=prompt_builder)
    
    query_service = QueryService(
        vector_store=vector_store,
        all_chunks=all_chunks,
        event_manager=api_event_manager,
        prompt_builder=prompt_builder,
        llm_client=llm_client
    )
    
    service_instances_cache[pdf_filename_basename] = (index_service, query_service)
    return index_service, query_service

# --- API Endpoints ---
@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    api_event_manager.emit('api_upload_request', {'filename': file.filename, 'content_type': file.content_type})
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # Size check
    max_size_bytes = settings.API_PDF_MAX_SIZE_MB * 1024 * 1024
    # Read the file content to check its size
    file_content = await file.read()
    file_size = len(file_content)
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size is {settings.API_PDF_MAX_SIZE_MB}MB. Provided: {file_size / (1024*1024):.2f}MB"
        )
    # Reset the file pointer for further reading
    file.file.seek(0)

    pdf_target_path = os.path.join(settings.PDFS_DIR, file.filename)
    
    try:
        # Save the uploaded PDF to the PDFS_DIR
        with open(pdf_target_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info(f"PDF '{file.filename}' uploaded and saved to '{pdf_target_path}'")
    except Exception as e:
        logger.error(f"Error saving uploaded PDF '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not save PDF: {str(e)}")
    finally:
        await file.close()

    try:
        # Initialize services with force_reindex=True for the uploaded PDF
        index_service, _ = await get_or_create_services(file.filename, force_reindex=True)
        status_message = "PDF processed and index created/updated successfully."
        index_status = "Indexed"
        if not index_service.get_vector_store() or not index_service.get_all_chunks():
            status_message = "PDF uploaded, but index creation might have issues. Check logs."
            index_status = "Indexing Error"
            logger.error(f"Index seems problematic for {file.filename} after processing upload.")
        return UploadResponse(
            message=status_message,
            pdf_filename=file.filename,
            index_status=index_status
        )
    except FileNotFoundError as fnf:
        logger.error(f"FileNotFoundError during upload processing for {file.filename}: {fnf}")
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        logger.error(f"Error processing PDF '{file.filename}' after upload: {e}", exc_info=True)
        if os.path.exists(pdf_target_path):
            logger.info(f"Uploaded PDF {pdf_target_path} kept despite processing error for debugging.")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


async def stream_answer_events(pdf_filename: str, question: str) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) for the /ask endpoint."""
    try:
        _, query_service = await get_or_create_services(pdf_filename, force_reindex=False)
    except FileNotFoundError:
        event_data = json.dumps({"error": f"PDF '{pdf_filename}' not found or not processed. Please upload it first."})
        yield f"event: error\ndata: {event_data}\n\n"
        return
    except Exception as e:
        logger.error(f"Failed to get services for {pdf_filename} during ask: {e}", exc_info=True)
        event_data = json.dumps({"error": f"Internal server error while preparing for your question: {str(e)}"})
        yield f"event: error\ndata: {event_data}\n\n"
        return

    try:
        async for event_type, data in query_service.answer_question_streaming(question):
            if event_type == "text_chunk":
                event_data = json.dumps({"chunk": data.get("chunk", "")})
                yield f"event: text_chunk\ndata: {event_data}\n\n"
            elif event_type == "sources":
                event_data = json.dumps({"sources": data.get("sources", [])})
                yield f"event: sources\ndata: {event_data}\n\n"
            elif event_type == "error": # If QueryService itself yields an error event
                event_data = json.dumps({"error": data.get("error", "An unknown error occurred during generation.")})
                yield f"event: error\ndata: {event_data}\n\n"
            await asyncio.sleep(0.01) # Small sleep to allow other tasks, adjust as needed
    except Exception as e:
        logger.error(f"Error during answer streaming for '{question}' on '{pdf_filename}': {e}", exc_info=True)
        event_data = json.dumps({"error": f"An error occurred while generating the answer: {str(e)}"})
        yield f"event: error\ndata: {event_data}\n\n"
    finally:
        # Signal end of stream (optional, client can also detect close)
        event_data = json.dumps({"message": "Stream ended."})
        yield f"event: end_stream\ndata: {event_data}\n\n"


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to handle user questions about a specific PDF.
    Emits events for logging and streams the response as Server-Sent Events (SSE).
    """
    api_event_manager.emit('api_ask_request', {'pdf_filename': request.pdf_filename, 'question': request.question})

    if not request.pdf_filename or not request.question:
        raise HTTPException(status_code=400, detail="pdf_filename and question are required.")

    # Stream the response using the helper function
    return StreamingResponse(
        stream_answer_events(request.pdf_filename, request.question),
        media_type="text/event-stream"
    )

# Example of a non-streaming endpoint (can be removed if only streaming is desired)
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = []
    cached_response: bool = False

@app.post("/ask-non-streaming", response_model=AnswerResponse)
async def ask_question_non_streaming(request: QuestionRequest):
    """
    Deprecated endpoint for non-streaming question answering.
    """
    api_event_manager.emit('api_ask_request_non_streaming', {'pdf_filename': request.pdf_filename, 'question': request.question})

    try:
        _, query_service = await get_or_create_services(request.pdf_filename, force_reindex=False)

        # Check cache first (QueryService handles its internal cache)
        cached_q = query_service.response_cache.get(request.question)
        if cached_q and isinstance(cached_q, dict) and 'final_answer' in cached_q:
            return AnswerResponse(
                answer=cached_q['final_answer'], 
                sources=cached_q['sources'], 
                cached_response=True
            )

        # Generate the answer and return it
        answer, sources = await query_service.answer_question_non_streaming(request.question)
        return AnswerResponse(answer=answer, sources=sources, cached_response=False)

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Generic error in /ask-non-streaming for {request.pdf_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    
    # Ensure directories from settings are created (settings.py does this on import)
    # os.makedirs(settings.PDFS_DIR, exist_ok=True)
    # os.makedirs(settings.INDICES_DIR, exist_ok=True)
    
    logger.info(f"Starting Uvicorn server on {settings.UVICORN_HOST}:{settings.UVICORN_PORT}")
    logger.info(f"PDFs will be stored in: {os.path.abspath(settings.PDFS_DIR)}")
    logger.info(f"Indices will be stored in: {os.path.abspath(settings.INDICES_DIR)}")
    logger.info(f"Ollama server expected at: {settings.OLLAMA_HOST}")
    
    uvicorn.run(
        "src.api.main:app", 
        host=settings.UVICORN_HOST, 
        port=settings.UVICORN_PORT, 
        reload=True, # For development
        timeout_keep_alive=settings.UVICORN_TIMEOUT_KEEP_ALIVE
    )
