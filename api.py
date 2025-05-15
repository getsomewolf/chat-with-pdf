# filepath: c:\Users\lucas\Projects\chat-with-pdf\api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from main import ChatWithPDF, PDFS_DIR, INDICES_DIR, EventManager, LoggingObserver, format_docs_for_api

app = FastAPI(title="Chat with PDF API", version="1.0")
chat_instances: dict[str, ChatWithPDF] = {}
api_event_manager = EventManager()
api_logger = LoggingObserver()
for ev in ['chat_with_pdf_initialized','retrieval_started','retrieval_completed','generation_started','generation_completed']:
    api_event_manager.subscribe(ev, api_logger)

class QuestionRequest(BaseModel):
    pdf_filename: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = []
    cached_response: bool = False


def get_chat_instance(pdf_filename: str) -> ChatWithPDF:
    if pdf_filename in chat_instances:
        return chat_instances[pdf_filename]
    pdf_path = os.path.join(PDFS_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        if not os.path.exists(pdf_filename):
            raise FileNotFoundError(f"PDF not found: {pdf_filename}")
        pdf_path = pdf_filename
    instance = ChatWithPDF(pdf_path, event_manager=api_event_manager)
    chat_instances[pdf_filename] = instance
    return instance


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        chat = get_chat_instance(request.pdf_filename)
        data = chat.ask_for_api(request.question)
        return AnswerResponse(answer=data['answer'], sources=data['sources'], cached_response=data.get('cached', False))
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    os.makedirs(PDFS_DIR, exist_ok=True)
    os.makedirs(INDICES_DIR, exist_ok=True)
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
