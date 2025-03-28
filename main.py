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

# Ignorar avisos para limpar a saída
warnings.filterwarnings("ignore")

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Definir diretórios do projeto
INDICES_DIR = "indices"
PDFS_DIR = "pdfs"
MODELS_DIR = "models"

# Criar diretórios se não existirem
for directory in [INDICES_DIR, PDFS_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório '{directory}' criado.")
        
# Função para formatar documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
        
#@lru_cache(maxsize=None)        
# Cache para armazenar índices de vectorstore previamente carregados
def get_vector_store(index_path, embeddings):
    #embeddings = HFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    """Carrega e armazena em cache os índices de vetor para reutilização rápida"""
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization= True)

# Cache para armazenar modelos LLM carregados
# temperature=0.1,                    max_tokens=256,                    top_p=0.85,                    n_ctx=4096,                    repeat_penalty=1.1,                    verbose=False,                    n_batch=256
#@lru_cache(maxsize=5)
def get_llm_model(model_path, temperature=0.1, max_tokens=512, top_p=0.95, n_ctx=4096, repeat_penalty=1.1):
    """Carrega e armazena em cache modelos LLM para reutilização rápida"""
    # Reduced batch size for faster loading
    # Added n_threads parameter for better CPU utilization
    # Decreased verbose to False to reduce console output
    return LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n_ctx=n_ctx,
        repeat_penalty=repeat_penalty,
        verbose=False,  # Changed from True to False to reduce console output
        n_batch=64,     # Reduced from 256 to 64 for faster loading
        n_threads=4     # Added to utilize multiple CPU cores efficiently
        
    )
    
def process_chunks_in_parallel(documents, text_splitter:RecursiveCharacterTextSplitter, max_workers=None):
    """
    Processa documentos em chunks paralelos para melhorar a performance.
    
    Args:
    documents: Lista de documentos a serem processados
    text_splitter: O divisor de texto a ser aplicado
    max_workers: Número máximo de workers (None= automático baseado na CPU)
    
    Returns:
        Lista de documentos divididos em chunks
    """
    
    # Para processar um único documento
    def process_document(document):
        return text_splitter.split_documents(document)
    
    # Processar documentos em paralelo
    chunks = []

    # Process all documents directly instead of parallel processing
    # which might be causing the tuple issue
    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        chunks.extend(split_docs)

    """  with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_document, documents))
        
    # Flatten os resultados
    for result in results:
        chunks.extend(result) """


        
    return chunks

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
        # Limpar a linha após parar a animação
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
        print("")  # Adicionado: nova linha para limpar a saída
        
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
        # Verificar se o arquivo existe
        if not os.path.exists(pdf_path):
            # Verificar se está na pasta pdfs
            pdf_in_dir = os.path.join(PDFS_DIR, os.path.basename(pdf_path))
            if os.path.exists(pdf_in_dir):
                pdf_path = pdf_in_dir
            else:
                raise ValueError(f"Arquivo não encontrado: {pdf_path}")
        
        # Se o arquivo não estiver na pasta de PDFs e existir, copie-o para lá
        if not pdf_path.startswith(PDFS_DIR):
            new_path = os.path.join(PDFS_DIR, os.path.basename(pdf_path))
            if not os.path.exists(new_path):
                shutil.copy2(pdf_path, new_path)
                print(f"PDF copiado para {new_path}")
            pdf_path = new_path
            
        self.pdf_path = pdf_path
        self.pdf_basename = os.path.basename(self.pdf_path).split('.')[0]
        self.index_path = os.path.join(INDICES_DIR, f"index_{self.pdf_basename}")
        self.response_cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # Cache para respostas frequentes
        self.qa_method = None  # Método detectado para consultar o qa
        self.setup()
    
    # Adicionado para verificar existência do índice
    def index_exists(self):
        return os.path.exists(self.index_path) and os.listdir(self.index_path)
    
    def determine_qa_method(self):
        """Determina qual método de consulta funciona corretamente com a instância qa."""
        # Using a very simple question to test
        test_question = "teste"
        
        # Try invoke first as it's most common in newer LangChain versions
        try:
            _ = self.qa.invoke({"input": test_question})
            print("Método 'invoke' detectado para processar perguntas.")
            self.qa_method = lambda q: self.qa.invoke({"input": q})
            return
        except Exception as e:
            print(f"Método 'invoke' falhou: {str(e)[:100]}...")  # Show only first 100 chars of error
        
        # Try direct call syntax
        try:
            _ = self.qa({"query": test_question})  # Changed from "question" to "query" which works in some versions
            print("Método 'call' detectado para processar perguntas.")
            self.qa_method = lambda q: self.qa({"query": q})
            return
        except Exception:
            try:
                _ = self.qa({"question": test_question})
                print("Método 'call' (question) detectado para processar perguntas.")
                self.qa_method = lambda q: self.qa({"question": q})
                return
            except Exception as e:
                print(f"Método 'call' falhou: {str(e)[:100]}...")
        
        # Try run method
        try:
            _ = self.qa.run(test_question)
            print("Método 'run' detectado para processar perguntas.")
            self.qa_method = lambda q: self.qa.run(q)
            return
        except Exception as e:
            print(f"Método 'run' falhou: {str(e)[:100]}...")
        
        # Enhanced fallback method with better error handling
        print("Usando método de fallback direto para processar perguntas.")
        def enhanced_fallback(q):
            try:
                # First try to get documents directly
                docs = self.retriever.get_relevant_documents(q)
                if not docs:
                    return {"result": "Não foram encontrados documentos relevantes para essa pergunta."}
                
                # Process each document to extract content
                contents = []
                for i, doc in enumerate(docs[:2]):  # Only use top 2 results to avoid overwhelming responses
                    try:
                        # Handle both Document objects and (Document, score) tuples
                        if isinstance(doc, tuple) and len(doc) >= 1:
                            content = doc[0].page_content if hasattr(doc[0], 'page_content') else str(doc[0])
                        elif hasattr(doc, "page_content"):
                            content = doc.page_content
                        else:
                            content = str(doc)
                        
                        contents.append(content)
                    except Exception as e:
                        print(f"Erro ao processar documento {i}: {e}")
                
                if not contents:
                    return {"result": "Não foi possível extrair conteúdo dos documentos recuperados."}
                
                # Simple response for now - just return the most relevant content
                return {"result": "Baseado no documento:\n\n" + contents[0]}
                
            except Exception as e:
                print(f"Erro no método fallback: {e}")
                return {"result": f"Ocorreu um erro ao processar a pergunta: {str(e)}"}
        
        self.qa_method = enhanced_fallback

    
    def setup(self):
        print(f"Preparando para processar: {self.pdf_path}")
        
        # Carregar embeddings
        with LoadingIndicator("Carregando embeddings") as loading:
            embeddings = HFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Verificar se já existe um índice para este PDF
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

                print(f"Document type: {type(documents)}")
                print(f"First document type: {type(documents[0]) if documents else 'No documents'}")
                print(f"Document structure: {documents[0].__dict__ if documents else 'No documents'}")
            
            # Dividir em chunks com sobreposição para melhor contexto
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=0,
                separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ""
                ]
            )            
            
            with LoadingIndicator("Dividindo documento em chunks") as loading:
                docs = process_chunks_in_parallel(documents, text_splitter)
                
            print(f"Criando embeddings e índice de pesquisa...")
            with LoadingIndicator("Criando vetores") as loading:
                self.vector_store = FAISS.from_documents(docs, embeddings)
                # Criar o diretório se não existir
                if not os.path.exists(self.index_path):
                    os.makedirs(self.index_path)
                self.vector_store.save_local(self.index_path)
            print(f"Índice criado e salvo em {self.index_path}")
        
        # Carregar o modelo LLM
        print("Carregando modelo de linguagem local...")
        
        # Caminho para o modelo
        str_model = "small"
        default_model_path = os.path.join(MODELS_DIR, f"ggml-model-{str_model}.bin")
        
        # Procurar por qualquer modelo disponível
        model_path = None
        if os.path.exists(default_model_path):
            model_path = default_model_path
        else:
            model_files = glob.glob(os.path.join(MODELS_DIR, "*.bin")) + glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
            if model_files:
                model_path = model_files[0]
                print(f"Usando modelo encontrado: {model_path}")
        
        if not model_path:
            print("\nATENÇÃO: Nenhum modelo encontrado!")
            print("Por favor, baixe um modelo GGML/GGUF (como o Llama-2-7B-Chat) e coloque-o no diretório 'models'")
            print("Você pode baixar modelos em: https://huggingface.co/TheBloke")
            print("Recomendado: Llama-2-7B-Chat-GGUF ou mistral-7B-instruct-v0.2.Q4_K_M.gguf")
            raise ValueError("Nenhum modelo LLM encontrado. Baixe um modelo e reinicie a aplicação.")
        
        # Template de prompt para respostas detalhadas
        prompt_template = """
            Você é um especialista em análise técnica de documentos. Sua resposta DEVE:
            1. Basear-se exclusivamente no conteúdo do PDF fornecido
            2. Usar estrutura clara com tópicos numerados
            3. Evitar qualquer inferência não explícita
            4. Indicar claramente quando informações são faltantes
            
            Contexto do documento: {context}
            Pergunta: {question}
            
            Resposta (formato livre, mas preferencialmente com marcas):
            [Conclusão Principal]
            [Detalhamento por tópicos]
            [Fontes no documento: página X]
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Configurando o LlamaCpp com o modelo local - com timeout
        try:
            # Run model loading with timeout to avoid hanging
            result, error = run_with_timeout(
                func=lambda: get_llm_model(model_path, max_tokens=256),  # Reduced max_tokens for faster responses
                timeout_duration=30  # Lower timeout for model loading
            )
            
            if error:
                raise ValueError(f"Tempo excedido ao carregar modelo: {error}")
                
            self.llm = result
            
            # Criar a cadeia de documentos (equivalente ao chain_type="stuff")
            combine_docs_chain = create_stuff_documents_chain(self.llm, PROMPT)
            
            # No método setup(), após criar o vector_store, adicione:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

            # Criação da cadeia LCEL
            self.qa = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | PROMPT
                | self.llm
                | StrOutputParser()
            )

            self.qa_method = lambda q: {"result": self.qa.invoke({"input": q})}
            
            print("Modelo local carregado e pronto para responder perguntas!")
            
            try:
                self.determine_qa_method()
            except Exception as e:
                print(f"Erro ao determinar método QA: {e}")
                # Definir método QA de fallback inline
                from functools import partial
                self.qa_method = partial(self._fallback_qa_method)

            
                
        except Exception as e:
            print(f"Erro ao carregar o modelo LlamaCpp: {e}")
            raise ValueError(f"Falha ao inicializar o modelo LLM: {e}")
        
    # E adicionar o método de fallback à classe:
    def _fallback_qa_method(self, question):
        docs = self.retriever.get_relevant_documents(question)
        if not docs:
            return {"result": "Não encontrei informações relevantes no documento."}
        
        context = docs[0].page_content
        # Usar o LLM diretamente
        self.response = self.llm(f"Documento: {context}\n\nPergunta: {question}\n\nResposta detalhada:")
        return {"result": self.response}


    def ask_optimized(self, question):
        """Versão otimizada do método ask com timeouts mais curtos e melhor tratamento de erros"""
        if not self.qa:
            raise ValueError("O sistema ainda não foi inicializado corretamente.")
        
        # Verificar cache
        if question in self.response_cache:
            print("Resposta encontrada no cache!")
            return self.response_cache[question]
        
        print(f"Processando pergunta: {question}")
        max_retries = 2
        
        for attempt in range(max_retries):
            # Usar timeout para evitar que o modelo fique preso
            with LoadingIndicator(f"Pensando sobre sua pergunta (tentativa {attempt+1}/{max_retries})") as loading:
                try:
                    # Reduced timeout
                    result, error = run_with_timeout(
                        func=lambda: self.qa_method(question),
                        timeout_duration=30  # Reduced from 120 to 30 seconds
                    )
                    
                    # If successful, break the retry loop
                    if not error:
                        break
                        
                    print(f"Tentativa {attempt+1} falhou com erro: {error}. " + 
                        ("Tentando novamente..." if attempt < max_retries-1 else ""))
                        
                except Exception as e:
                    print(f"Erro ao processar a pergunta: {e}")
                    error = e
        
        # If we have an error after all retries
        if error and isinstance(error, TimeoutError):
            # Fallback to direct document retrieval
            try:
                docs = self.retriever
                if docs:
                    response = f"Não foi possível obter uma resposta completa, mas encontrei este trecho relevante:\n\n{docs[0].page_content}"
                else:
                    response = f"Não foi possível obter uma resposta: {str(error)}"
            except Exception:
                response = f"Não foi possível obter uma resposta: {str(error)}"
                
            self.response_cache[question] = response
            return response
        
        # Extrair o resultado (formato mudou)
        if isinstance(result, dict):
            if "answer" in result:
                response = result["answer"]
            elif "result" in result:
                response = result["result"]
            else:
                response = str(result)
        elif isinstance(result, str):
            response = result
        else:
            # Fallback para outros formatos
            response = str(result)
        
        # Adicionar verificação para evitar respostas simples
        if len(response.split()) < 20 and question.lower() not in ["teste", "test"]:
            print("DEBUG: Resposta muito curta, forçando processamento pelo LLM")
            try:
                # Force LLM processing
                context = self.retriever.get_relevant_documents(question)[0].page_content
                response = self.llm(f"Contexto do documento PDF: {context}\n\nPergunta: {question}\n\nForneça uma resposta detalhada:")
            except Exception as e:
                print(f"DEBUG: Erro ao forçar processamento: {e}")
            
        # Armazenar no cache
        self.response_cache[question] = response
        
        return response

def list_available_pdfs():
    """Lista todos os PDFs disponíveis na pasta pdfs e no diretório atual."""
    # PDFs na pasta dedicada
    pdfs_in_dir = [os.path.join(PDFS_DIR, f) for f in os.listdir(PDFS_DIR) if f.lower().endswith('.pdf')]
    
    # PDFs no diretório atual (que não estejam na pasta pdfs)
    pdfs_in_current = [f for f in glob.glob("*.pdf") if not f.startswith(PDFS_DIR)]
    
    # Combinar as listas, priorizando os da pasta pdfs
    all_pdfs = pdfs_in_dir + pdfs_in_current
    
    return all_pdfs

def has_index(pdf_path):
    """Verifica se um PDF já possui índice criado."""
    basename = os.path.basename(pdf_path).split('.')[0]
    index_path = os.path.join(INDICES_DIR, f"index_{basename}")
    return os.path.exists(index_path) and len(os.listdir(index_path)) > 0

def select_pdf():
    """Permite ao usuário selecionar um PDF dentre os disponíveis."""
    all_pdfs = list_available_pdfs()
    
    if not all_pdfs:
        print("\nNenhum PDF encontrado no sistema.")
        pdf_path = input("Digite o caminho completo para um arquivo PDF: ")
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        return pdf_path
    
    # Mostrar os PDFs disponíveis, indicando quais já possuem índice
    print("\nPDFs disponíveis:")
    for i, pdf in enumerate(all_pdfs, 1):
        indexed = " [indexado]" if has_index(pdf) else ""
        print(f"{i}. {os.path.basename(pdf)}{indexed}")
    
    # Permitir que o usuário selecione um PDF ou digite um novo caminho
    choice = input("\nDigite o número do PDF ou o caminho completo para um novo arquivo: ")
    
    # Se for um número, seleciona da lista
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(all_pdfs):
            return all_pdfs[index]
        else:
            print("Número inválido.")
            return None
    # Se não for um número, assume que é um caminho
    elif choice.strip():
        if os.path.exists(choice):
            return choice
        else:
            print(f"Arquivo não encontrado: {choice}")
            return None
    # Se estiver vazio, retorna None
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
    """Remove índices para PDFs que não existem mais."""
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
    
    # Limpar índices não utilizados
    cleanup_unused_indices()
    
    # Permitir que o usuário selecione um PDF
    pdf_path = select_pdf()
    
    if not pdf_path:
        print("\nNenhum PDF selecionado. Encerrando o programa.")
        exit()
        
    try:
        chat = ChatWithPDF(pdf_path)
        
        # Interface de chat
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
