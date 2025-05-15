import pytest
import asyncio
from fastapi.testclient import TestClient
import os
import shutil

# Import app from api.py. Adjust if your app instance is named differently or located elsewhere.
# This assumes your FastAPI app instance is named `app` in `api.py` at the project root.
# To make this work, ensure your project root is in PYTHONPATH when running tests,
# or use relative imports if your test structure allows.
# For simplicity, we'll assume api.py is discoverable.
from api import app as fastapi_app # api.py needs to be in PYTHONPATH
from config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """
    FastAPI TestClient fixture.
    This client will be used to make requests to your API endpoints.
    """
    # Ensure test-specific directories exist and are clean
    test_pdfs_dir = os.path.join(settings.PDFS_DIR, "test_data")
    test_indices_dir = os.path.join(settings.INDICES_DIR, "test_data")
    
    # Override settings for testing if necessary, e.g., point to test dirs
    # For now, we assume default settings are okay, or tests manage their own files.
    # If you modify settings, ensure it's done safely (e.g., via environment variables for tests)

    os.makedirs(test_pdfs_dir, exist_ok=True)
    os.makedirs(test_indices_dir, exist_ok=True)
    
    # Provide a client
    client = TestClient(fastapi_app)
    yield client

    # Teardown: Clean up test directories after tests in the module are done
    # Be careful with this if tests run in parallel or if you want to inspect output
    # shutil.rmtree(test_pdfs_dir, ignore_errors=True)
    # shutil.rmtree(test_indices_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_pdf_file(tmp_path_factory):
    """Creates a temporary dummy PDF file for upload tests."""
    # tmp_path_factory is a pytest fixture that provides a temporary directory unique to the test invocation
    fn = tmp_path_factory.mktemp("data") / "test_document.pdf"
    # Create a minimal valid PDF structure or copy a small fixture PDF
    # For simplicity, creating a text file and naming it .pdf
    # For real tests, use a tiny valid PDF.
    with open(fn, "w") as f:
        f.write("%PDF-1.4\n%âãÏÓ\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000059 00000 n\n0000000118 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF")
    return fn

@pytest.fixture(scope="module")
def mock_ollama_client_success(mocker):
    """Mocks ollama.AsyncClient to simulate successful LLM responses."""
    mock_async_client = mocker.MagicMock(spec=ollama.AsyncClient) # type: ignore

    async def mock_chat_stream(*args, **kwargs):
        yield {"message": {"content": "Mocked LLM response chunk 1."}}
        await asyncio.sleep(0.01)
        yield {"message": {"content": " Mocked LLM response chunk 2."}}

    async def mock_chat_non_stream(*args, **kwargs):
        return {"message": {"content": "Mocked full LLM response."}}

    if 'stream' in mocker.patch.object(ollama.AsyncClient, 'chat').call_args.kwargs and \
       mocker.patch.object(ollama.AsyncClient, 'chat').call_args.kwargs['stream'] is True: # type: ignore
        mock_async_client.chat = mock_chat_stream
    else:
        mock_async_client.chat = mock_chat_non_stream
        
    # Patch the constructor of LLMClient to use this mock
    # This is a bit broad. Better to patch ollama.AsyncClient where it's instantiated.
    # For now, let's assume LLMClient gets it via DI or direct instantiation.
    # We will mock `ollama.AsyncClient` directly when LLMClient is created.
    
    # This fixture provides the mock_async_client instance.
    # Tests will need to ensure LLMClient uses this.
    # One way: patch('llm_client.ollama.AsyncClient', return_value=mock_async_client)
    return mock_async_client

# Add more fixtures as needed, e.g., for pre-populating PDFS_DIR/INDICES_DIR for certain tests.
