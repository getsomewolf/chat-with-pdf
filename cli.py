import os
import glob
import warnings
from datetime import datetime
from event_manager import EventManager, Observer
from services import IndexService, QueryService, PromptBuilder, LLMClient
from utils.cli_utils import LoadingIndicator
from utils.file_utils import list_available_pdfs, has_index, select_pdf_cli, cleanup_unused_indices_cli
from config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

# Ignorar avisos para limpar a saída
warnings.filterwarnings("ignore")

# Definir diretórios do projeto
PDFS_DIR = settings.PDFS_DIR
INDICES_DIR = settings.INDICES_DIR

# Criar diretórios se não existirem
for directory in [INDICES_DIR, PDFS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório '{directory}' criado.")

def print_header():
    print("=" * 70)
    print(f"{'CHAT WITH PDF - VERSÃO 3.5':^70}")
    print(f"{'Modelo Local com Respostas Detalhadas':^70}")
    print(f"{'Data: ' + datetime.now().strftime('%d/%m/%Y %H:%M'):^70}")
    print("=" * 70)

def print_help():
    print("\nSugestões de perguntas para respostas detalhadas:")
    print("- Qual é o tema principal deste documento?")
    print("- Resuma as informações mais importantes deste PDF.")
    print("- Explique detalhadamente o que este documento aborda sobre [tópico].")
    print("- Quais são os principais pontos discutidos na seção [X]?")
    print("- Como o documento relaciona [conceito A] com [conceito B]?")
    print("- Extraia todas as informações técnicas sobre [assunto].")

class CLIObserver(Observer):
    def update(self, event_type: str, data: dict = None):
        print(f"[EVENT] {event_type}: {data}")

async def main_cli():
    print_header()
    cleanup_unused_indices_cli()

    pdf_path = select_pdf_cli()
    if not pdf_path:
        print("\nNenhum PDF selecionado. Encerrando o programa.")
        return

    event_manager = EventManager()
    cli_observer = CLIObserver()
    for event in [
        'index_setup_started', 'index_loaded', 'index_creation_started',
        'chunks_split', 'index_created', 'index_setup_completed',
        'retrieval_started', 'retrieval_completed', 'generation_started',
        'generation_completed'
    ]:
        event_manager.subscribe(event, cli_observer)

    index_service = IndexService(pdf_path, event_manager=event_manager)
    query_service = None

    try:
        await index_service.initialize_index(use_cli_indicator=True)
        vector_store = index_service.get_vector_store()
        all_chunks = index_service.get_all_chunks()

        if not vector_store or not all_chunks:
            print("Erro: Não foi possível carregar o índice ou os chunks.")
            return

        query_service = QueryService(
            vector_store=vector_store,
            all_chunks=all_chunks,
            event_manager=event_manager,
            prompt_builder=PromptBuilder(),  # Instanciação correta do PromptBuilder
            llm_client=LLMClient(event_manager=event_manager, prompt_builder=PromptBuilder())  # Instanciação correta do LLMClient
        )
    except Exception as e:
        logger.error(f"Erro ao inicializar serviços: {e}")
        print("Erro ao inicializar os serviços. Verifique os logs para mais detalhes.")
        return

    while True:
        user_question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if user_question.lower() in ["sair", "exit", "quit"]:
            print("\nEncerrando o programa. Até mais!")
            break

        try:
            async for event_type, data in query_service.answer_question_streaming(user_question):
                if event_type == "text_chunk":
                    print(data["chunk"], end="", flush=True)
                elif event_type == "sources":
                    print("\n\nFontes:")
                    for source in data["sources"]:
                        print(f"- {source}")
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            print("Erro ao processar a pergunta. Verifique os logs para mais detalhes.")

if __name__ == "__main__":
    asyncio.run(main_cli())
