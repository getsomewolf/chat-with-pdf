# LLM client for interacting with Ollama
import time
import ollama # type: ignore
from event_manager import EventManager
from prompt_builder import PromptBuilder
from config import settings
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, event_manager: EventManager, prompt_builder: PromptBuilder):
        self.model_name = settings.OLLAMA_MODEL_NAME
        self.event_manager = event_manager
        self.prompt_builder = prompt_builder
        # Initialize AsyncClient with host and timeout
        self.async_client = ollama.AsyncClient(
            host=settings.OLLAMA_HOST, 
            timeout=settings.OLLAMA_TIMEOUT
        )
        logger.info(f"LLMClient initialized for model: {self.model_name} at host: {settings.OLLAMA_HOST}")

    async def generate(self, context: str, question: str) -> AsyncGenerator[str, None]:
        """
        Envia prompt para o modelo e retorna um gerador assíncrono para a resposta em chunks.
        """
        self.event_manager.emit('generation_started', {'question': question, 'model': self.model_name})
        start_time = time.time()
        
        prompt = self.prompt_builder.build(context, question)
        
        try:
            stream = await self.async_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            
            full_answer = ""
            async for chunk in stream:
                content_part = chunk['message']['content']
                full_answer += content_part
                yield content_part
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            self.event_manager.emit('generation_failed', {'error': str(e)})
            # Yield an error message or re-raise, depending on desired handling
            yield f"Error communicating with LLM: {str(e)}" # Or raise e
            return # Ensure generator stops

        elapsed_time = time.time() - start_time
        self.event_manager.emit('generation_completed', {'time': elapsed_time, 'answer_length': len(full_answer)})
        logger.info(f"LLM generation completed in {elapsed_time:.2f}s")

    async def generate_non_streaming(self, context: str, question: str) -> str:
        """
        Envia prompt para o modelo e retorna a resposta completa (não-streaming).
        """
        self.event_manager.emit('generation_started', {'question': question, 'model': self.model_name, 'streaming': False})
        start_time = time.time()
        prompt = self.prompt_builder.build(context, question)
        
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False # Explicitly non-streaming
            )
            answer = response['message']['content']
        except Exception as e:
            logger.error(f"Error during non-streaming LLM generation: {e}")
            self.event_manager.emit('generation_failed', {'error': str(e)})
            raise  # Re-raise the exception to be handled by the caller

        elapsed_time = time.time() - start_time
        self.event_manager.emit('generation_completed', {'time': elapsed_time, 'answer_length': len(answer), 'streaming': False})
        logger.info(f"Non-streaming LLM generation completed in {elapsed_time:.2f}s")
        return answer
