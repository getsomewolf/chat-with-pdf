from dotenv import load_dotenv
import os
import glob
import time
import warnings
import threading
import sys
from datetime import datetime
import shutil
import cachetools
import ollama


# Ignorar avisos para limpar a saída
warnings.filterwarnings("ignore")

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
from langchain_community.vectorstores import FAISS
from functools import lru_cache # não usado, mas pode ser útil para otimização futura




# Definir diretórios do projeto
INDICES_DIR = "indices"
PDFS_DIR = "pdfs"

# Criar diretórios se não existirem
for directory in [INDICES_DIR, PDFS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório '{directory}' criado.")

# Função para formatar documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cache para armazenar índices de vectorstore previamente carregados
_vector_store_cache = {}
def get_vector_store(index_path, embeddings):
    if index_path not in _vector_store_cache:
        _vector_store_cache[index_path] = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return _vector_store_cache[index_path]
# Classe para mostrar animação de loading
class LoadingIndicator:
    def __init__(self, message="Processando"):
        self.message = message
        self.is_running = False
        self.animation_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.is_running = True
        self.animation_thread = threading.Thread(target=self._animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop(self):
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
        print("")  # Nova linha para limpar a saída

    def _animate(self):
        animation = "|/-\\"
        idx = 0
        while self.is_running:
            progress = animation[idx % len(animation)]
            sys.stdout.write(f'\r{self.message} {progress}')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

# Função para executar com timeout
def run_with_timeout(func, args=(), kwargs={}, timeout_duration=120):
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)

    if thread.is_alive():
        return None, TimeoutError(f"A operação excedeu o limite de tempo de {timeout_duration} segundos")

    if error[0]:
        return None, error[0]

    return result[0], None

class ChatWithPDF:
    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            pdf_in_dir = os.path.join(PDFS_DIR, os.path.basename(pdf_path))
            if os.path.exists(pdf_in_dir):
                pdf_path = pdf_in_dir
            else:
                raise ValueError(f"Arquivo não encontrado: {pdf_path}")

        if not pdf_path.startswith(PDFS_DIR):
            new_path = os.path.join(PDFS_DIR, os.path.basename(pdf_path))
            if not os.path.exists(new_path):
                shutil.copy2(pdf_path, new_path)
                print(f"PDF copiado para {new_path}")
            pdf_path = new_path

        self.pdf_path = pdf_path
        self.pdf_basename = os.path.basename(self.pdf_path).split('.')[0]
        self.index_path = os.path.join(INDICES_DIR, f"index_{self.pdf_basename}")
        self.response_cache = cachetools.TTLCache(maxsize=100, ttl=3600)
        self.setup()

    def index_exists(self):
        return os.path.exists(self.index_path) and os.listdir(self.index_path)

    def setup(self):
        print(f"Preparando para processar: {self.pdf_path}")

        with LoadingIndicator("Carregando embeddings") as loading:
            embeddings = HFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if self.index_exists():
            print(f"Índice encontrado para {self.pdf_path}. Carregando índice existente...")
            with LoadingIndicator("Carregando índice") as loading:
                self.vector_store = get_vector_store(self.index_path, embeddings)
            print("Índice carregado com sucesso!")
        else:
            print(f"Processando o PDF e criando novo índice...")
            with LoadingIndicator("Lendo PDF") as loading:
                loader = PyPDFLoader(file_path=self.pdf_path)
                documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
            )

            with LoadingIndicator("Dividindo documento em chunks") as loading:
                docs = text_splitter.split_documents(documents)

            print(f"Criando embeddings e índice de pesquisa...")
            with LoadingIndicator("Criando vetores") as loading:
                self.vector_store = FAISS.from_documents(docs, embeddings)
                if not os.path.exists(self.index_path):
                    os.makedirs(self.index_path)
                self.vector_store.save_local(self.index_path)
            print(f"Índice criado e salvo em {self.index_path}")

        # Configurar o retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

    def ask_optimized(self, question):
        if question in self.response_cache:
            print("Resposta encontrada no cache!")
            return self.response_cache[question]

        print(f"Processando pergunta: {question}")
        max_retries = 2

        for attempt in range(max_retries):
            with LoadingIndicator(f"Pensando sobre sua pergunta (tentativa {attempt+1}/{max_retries})") as loading:
                try:
                    # Recuperar documentos relevantes
                    docs = self.retriever.get_relevant_documents(question)
                    if not docs:
                        return "Não foram encontrados documentos relevantes para essa pergunta."

                    context = format_docs(docs)

                    # Usar Ollama para gerar a resposta
                    response = ollama.chat(model="llama3.2", messages=[
                        {
                            'role': 'user',
                            'content': f"Contexto: {context}\n\nPergunta: {question}\n\nResposta detalhada:"
                        },
                    ])
                    

                    # Extrair a resposta
                    answer = response['message']['content']
                    self.response_cache[question] = answer
                    return answer

                except Exception as e:
                    print(f"Tentativa {attempt+1} falhou com erro: {e}")
                    if attempt == max_retries - 1:
                        return f"Erro ao processar a pergunta: {e}"

def list_available_pdfs():
    pdfs_in_dir = [os.path.join(PDFS_DIR, f) for f in os.listdir(PDFS_DIR) if f.lower().endswith('.pdf')]
    pdfs_in_current = [f for f in glob.glob("*.pdf") if not f.startswith(PDFS_DIR)]
    all_pdfs = pdfs_in_dir + pdfs_in_current
    return all_pdfs

def has_index(pdf_path):
    basename = os.path.basename(pdf_path).split('.')[0]
    index_path = os.path.join(INDICES_DIR, f"index_{basename}")
    return os.path.exists(index_path) and len(os.listdir(index_path)) > 0

def select_pdf():
    all_pdfs = list_available_pdfs()

    if not all_pdfs:
        print("\nNenhum PDF encontrado no sistema.")
        pdf_path = input("Digite o caminho completo para um arquivo PDF: ")
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        return pdf_path

    print("\nPDFs disponíveis:")
    for i, pdf in enumerate(all_pdfs, 1):
        indexed = " [indexado]" if has_index(pdf) else ""
        print(f"{i}. {os.path.basename(pdf)}{indexed}")

    choice = input("\nDigite o número do PDF ou o caminho completo para um novo arquivo: ")

    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(all_pdfs):
            return all_pdfs[index]
        else:
            print("Número inválido.")
            return None
    elif choice.strip():
        if os.path.exists(choice):
            return choice
        else:
            print(f"Arquivo não encontrado: {choice}")
            return None
    else:
        return None

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

def cleanup_unused_indices():
    if not os.path.exists(INDICES_DIR):
        return

    for index_dir in os.listdir(INDICES_DIR):
        if index_dir.startswith("index_"):
            pdf_name = index_dir.replace("index_", "") + ".pdf"
            pdf_path = os.path.join(PDFS_DIR, pdf_name)

            if not os.path.exists(pdf_path):
                index_path = os.path.join(INDICES_DIR, index_dir)
                print(f"Removendo índice para PDF não encontrado: {pdf_name}")
                try:
                    shutil.rmtree(index_path)
                except Exception as e:
                    print(f"Erro ao remover índice: {e}")

if __name__ == "__main__":
    print_header()

    cleanup_unused_indices()

    pdf_path = select_pdf()

    if not pdf_path:
        print("\nNenhum PDF selecionado. Encerrando o programa.")
        exit()

    try:
        chat = ChatWithPDF(pdf_path)

        print("\n" + "=" * 70)
        print(f"{'MODO DE CHAT - RESPOSTAS DETALHADAS':^70}")
        print("=" * 70)
        print("Digite suas perguntas sobre o documento para obter informações detalhadas.")
        print("Digite 'sair', 'exit' ou 'quit' para finalizar.")
        print("Digite 'ajuda' ou 'help' para ver sugestões de perguntas.")

        while True:
            user_question = input("\nPergunta: ")
            question_lower = user_question.lower()

            if question_lower in ["sair", "quit", "exit"]:
                print("\nEncerrando chat. Até mais!")
                break

            if not question_lower.strip():
                print("\nPor favor, digite uma pergunta.")
                continue

            if question_lower in ["ajuda", "help"]:
                print_help()
                continue

            start_time = time.time()
            answer = chat.ask_optimized(user_question)
            elapsed_time = time.time() - start_time

            print(f"\nResposta ({elapsed_time:.1f}s):", answer)
    except Exception as e:
        print(f"\nErro: {e}")
        print("Tente novamente com um arquivo PDF válido.")