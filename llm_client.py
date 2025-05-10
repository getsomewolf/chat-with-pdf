# LLM client for interacting with Ollama
import time
import ollama
from event_manager import EventManager
from prompt_builder import PromptBuilder

class LLMClient:
    def __init__(self, model_name: str, event_manager: EventManager, prompt_builder: PromptBuilder):
        self.model_name = model_name
        self.event_manager = event_manager
        self.prompt_builder = prompt_builder

    def generate(self, context: str, question: str) -> str:
        """
        Envia prompt para o modelo e retorna resposta completa.
        """
        self.event_manager.emit('generation_started', {'question': question})
        start = time.time()
        # construir prompt com PromptBuilder
        prompt = self.prompt_builder.build(context, question)
        # enviar à API do Ollama
        stream = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        answer = ""
        for chunk in stream:
            answer += chunk['message']['content']
        elapsed = time.time() - start
        self.event_manager.emit('generation_completed', {'time': elapsed})
        return answer
