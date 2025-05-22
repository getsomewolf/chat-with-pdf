import pytest
from fastapi.testclient import TestClient
import os
import shutil
import json
import asyncio

from src.config.settings import settings # For accessing PDFS_DIR, INDICES_DIR

# Assuming conftest.py provides test_client and temp_pdf_file fixtures
# pytestmark = pytest.mark.asyncio # if using async test functions directly

def test_upload_pdf_success(test_client: TestClient, temp_pdf_file: str):
    """Test successful PDF upload and basic indexing trigger."""
    # Clean up any existing index for this test file to ensure it's created
    pdf_basename = os.path.basename(temp_pdf_file)
    index_name = pdf_basename.split('.')[0]
    test_index_path = os.path.join(settings.INDICES_DIR, f"index_{index_name}")
    if os.path.exists(test_index_path):
        shutil.rmtree(test_index_path)
    
    # Ensure the target PDF dir is clean for this specific file
    target_pdf_in_managed_dir = os.path.join(settings.PDFS_DIR, pdf_basename)
    if os.path.exists(target_pdf_in_managed_dir):
        os.remove(target_pdf_in_managed_dir)

    with open(temp_pdf_file, "rb") as f:
        response = test_client.post("/upload-pdf/", files={"file": (pdf_basename, f, "application/pdf")})

    assert response.status_code == 200
    data = response.json()
    assert data["pdf_filename"] == pdf_basename
    assert data["message"] == "PDF processed and index created/updated successfully." # Or similar success message
    assert data["index_status"] == "Indexed"

    # Verify PDF is in PDFS_DIR
    assert os.path.exists(target_pdf_in_managed_dir)
    # Verify index directory is created (basic check)
    assert os.path.exists(test_index_path)
    assert len(os.listdir(test_index_path)) > 0 # FAISS creates index.faiss and index.pkl

    # Clean up created files for this test
    if os.path.exists(target_pdf_in_managed_dir):
        os.remove(target_pdf_in_managed_dir)
    if os.path.exists(test_index_path):
        shutil.rmtree(test_index_path)


def test_upload_pdf_too_large(test_client: TestClient, tmp_path):
    """Test PDF upload exceeding size limit."""
    large_file_path = tmp_path / "large_file.pdf"
    # Create a dummy file larger than the configured max size
    # settings.API_PDF_MAX_SIZE_MB is in MB
    max_size_bytes = settings.API_PDF_MAX_SIZE_MB * 1024 * 1024
    with open(large_file_path, "wb") as f:
        f.write(b"0" * (max_size_bytes + 100)) # 100 bytes over limit

    with open(large_file_path, "rb") as f:
        response = test_client.post("/upload-pdf/", files={"file": ("large_file.pdf", f, "application/pdf")})
    
    assert response.status_code == 413 # Payload Too Large
    data = response.json()
    assert "File too large" in data["detail"]


@pytest.mark.asyncio # Mark test as async if it uses await directly or needs an event loop
async def test_ask_streaming_success(test_client: TestClient, temp_pdf_file: str, mocker):
    """Test /ask endpoint with streaming response after a PDF is uploaded."""
    # 1. Upload a PDF first to ensure it's processed and indexed.
    pdf_basename = os.path.basename(temp_pdf_file)
    index_name = pdf_basename.split('.')[0]
    test_index_path = os.path.join(settings.INDICES_DIR, f"index_{index_name}")
    target_pdf_in_managed_dir = os.path.join(settings.PDFS_DIR, pdf_basename)

    # Clean up from previous runs if any
    if os.path.exists(test_index_path): shutil.rmtree(test_index_path)
    if os.path.exists(target_pdf_in_managed_dir): os.remove(target_pdf_in_managed_dir)

    with open(temp_pdf_file, "rb") as f_upload:
        upload_response = test_client.post("/upload-pdf/", files={"file": (pdf_basename, f_upload, "application/pdf")})
    assert upload_response.status_code == 200
    
    # Mock the LLMClient's generate method within QueryService for this test
    # to avoid actual Ollama calls and control the output.
    # This is a bit tricky as QueryService is instantiated dynamically.
    # A better way might be to have a dependency injection system for LLMClient in QueryService
    # or mock ollama.AsyncClient.chat directly.
    
    # For this example, let's mock at the ollama.AsyncClient.chat level
    # This mock will apply to the LLMClient instance used by the QueryService for this PDF.
    mock_chat_stream = mocker.AsyncMock()
    async def mock_ollama_chat_stream_gen(*args, **kwargs):
        yield {"message": {"content": "Streamed part 1 from mock. "}}
        await asyncio.sleep(0.01) # Simulate network delay
        yield {"message": {"content": "Streamed part 2 from mock."}}
    mock_chat_stream.side_effect = mock_ollama_chat_stream_gen
    
    # Patch where AsyncClient is instantiated or its `chat` method
    # Assuming LLMClient uses ollama.AsyncClient().chat
    # The actual path to patch depends on how LLMClient is structured.
    # If LLMClient has an `self.async_client` attribute:
    mocker.patch('ollama.AsyncClient.chat', new=mock_chat_stream)
    # If you have `from ollama import AsyncClient` in llm_client.py, then:
    # mocker.patch('llm_client.AsyncClient.chat', new=mock_chat_stream)

    # 2. Ask a question
    request_data = {"pdf_filename": pdf_basename, "question": "What is in this document?"}
    
    # Make the streaming request
    # TestClient handles streaming responses by iterating over response.iter_lines() or similar
    # For SSE, we expect lines like "event: <type>\ndata: <json_payload>\n\n"
    
    received_events = []
    with test_client.stream("POST", "/ask", json=request_data) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        current_event = {}
        for line_bytes in response.iter_lines(): # iter_lines gives bytes
            line = line_bytes.decode('utf-8')
            if line.startswith("event:"):
                if current_event.get("type") and current_event.get("data"): # Store previous complete event
                    received_events.append(current_event)
                    current_event = {}
                current_event["type"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                current_event["data"] = json.loads(line.split(":", 1)[1].strip())
            elif not line.strip() and current_event: # Empty line signifies end of an event
                if current_event.get("type") and "data" in current_event: # Ensure data key exists, even if None
                     received_events.append(current_event)
                current_event = {}
        
        if current_event.get("type") and "data" in current_event : # Catch last event if no trailing newline
            received_events.append(current_event)


    # Assertions on received events
    assert any(event["type"] == "sources" for event in received_events)
    text_chunks = [event["data"]["chunk"] for event in received_events if event["type"] == "text_chunk"]
    assert "Streamed part 1 from mock. " in "".join(text_chunks)
    assert "Streamed part 2 from mock." in "".join(text_chunks)
    assert any(event["type"] == "end_stream" for event in received_events)

    # Verify the mock was called
    mock_chat_stream.assert_called()

    # Clean up
    if os.path.exists(target_pdf_in_managed_dir): os.remove(target_pdf_in_managed_dir)
    if os.path.exists(test_index_path): shutil.rmtree(test_index_path)

def test_ask_pdf_not_found(test_client: TestClient):
    """Test /ask endpoint when PDF does not exist."""
    request_data = {"pdf_filename": "non_existent_document.pdf", "question": "Hello?"}
    
    # Expecting an error event in the stream
    received_events = []
    with test_client.stream("POST", "/ask", json=request_data) as response:
        assert response.status_code == 200 # Stream itself is 200, error is in events
        current_event = {}
        for line_bytes in response.iter_lines():
            line = line_bytes.decode('utf-8')
            if line.startswith("event:"):
                if current_event: received_events.append(current_event)
                current_event = {"type": line.split(":", 1)[1].strip()}
            elif line.startswith("data:"):
                current_event["data"] = json.loads(line.split(":", 1)[1].strip())
            elif not line.strip() and current_event:
                 received_events.append(current_event)
                 current_event = {}
        if current_event: received_events.append(current_event)

    error_event = next((event for event in received_events if event["type"] == "error"), None)
    assert error_event is not None
    assert "PDF 'non_existent_document.pdf' not found" in error_event["data"]["error"]
    assert any(event["type"] == "end_stream" for event in received_events)

# Add more tests:
# - /ask with LLM failure (mock LLMClient to raise an exception)
# - /ask-non-streaming (if keeping it): success, PDF not found, caching
# - Test specific error codes and messages for various failure scenarios in /upload-pdf/
