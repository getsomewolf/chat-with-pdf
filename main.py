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
import re


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
    formatted_text = ""
    print(f"Formatando {len(docs)} documentos recuperados:")
    
    for i, doc in enumerate(docs):
        # Adicionar metadados como fonte se disponíveis
        source_info = ""
        if doc.metadata and 'source' in doc.metadata:
            source_path = doc.metadata['source']
            page_info = f", Página {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
            source_info = f"[Fonte: {os.path.basename(source_path)}{page_info}]"
        
        # Adicionar índice do chunk e metadados ao log
        chunk_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"Chunk {i+1}/{len(docs)} {source_info}:\n{chunk_preview}\n")
        
        # Adicionar separador claro entre chunks para o texto formatado
        formatted_text += f"{doc.page_content}\n"
        if source_info:
            formatted_text += f"{source_info}\n"
        formatted_text += "\n" + "-" * 40 + "\n"
    
    return formatted_text

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
        # Clear the entire line to ensure no loading animation characters remain
        sys.stdout.write('\r' + ' ' * (len(self.message) + 50) + '\r')
        sys.stdout.flush()

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
        
        # Configurações de chunking melhoradas
        self.chunk_size = 1000       # Tamanho equilibrado para capturar contexto significativo
        self.chunk_overlap = 200     # Overlap aumentado para manter contexto entre chunks
        
        # Configurações de recuperação melhoradas
        self.retrieval_k = 4         # Aumentado para capturar mais contexto potencialmente relevante
        self.diversity_lambda = 0.25  # Ligeiramente ajustado para favorecer relevância com diversidade
        
        # Configuração para override manual quando necessário
        self.force_reindex = False
        
        # Inicialização
        self.setup()

    def index_exists(self):
        return os.path.exists(self.index_path) and os.listdir(self.index_path)

    def setup(self):
        print(f"Preparando para processar: {self.pdf_path}")

        with LoadingIndicator("Carregando embeddings") as loading:
            # Modelo de embeddings mais robusto para melhor captura semântica
            # Usando um modelo melhor para captura semântica mais precisa
            embeddings = HFEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2", 
                show_progress=True
            )
            print(f"Modelo de embeddings carregado: sentence-transformers/all-mpnet-base-v2")

        if self.index_exists() and not self.force_reindex:
            print(f"Índice encontrado para {self.pdf_path}. Carregando índice existente...")
            with LoadingIndicator("Carregando índice") as loading:
                self.vector_store = get_vector_store(self.index_path, embeddings)
            print(f"Índice carregado com sucesso: {self.index_path}")
            
            # Verificar a integridade do índice
            try:
                test_query = "teste"
                docs = self.vector_store.similarity_search(test_query, k=1)
                print(f"Índice verificado e funcional - encontrado {len(docs)} documento(s) de teste")
            except Exception as e:
                print(f"AVISO: Teste de índice falhou: {e}")
                print("Recriando índice para garantir integridade...")
                self._create_index(embeddings)
        else:
            if self.force_reindex and self.index_exists():
                print(f"Forçando recriação do índice conforme solicitado.")
            else:
                print(f"Nenhum índice encontrado. Processando o PDF e criando novo índice...")
            self._create_index(embeddings)

        # Configurar o retriever com parâmetros aprimorados
        print(f"Configurando retriever aprimorado com k={self.retrieval_k} e lambda_mult={self.diversity_lambda}...")
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Usar Maximum Marginal Relevance para melhor diversidade
            search_kwargs={
                'k': self.retrieval_k,
                'lambda_mult': self.diversity_lambda,  # Ajuste para diversidade                
                'fetch_k': self.retrieval_k * 2  # Buscar mais candidatos para selecionar os mais diversos
            }
        )
        print("Retriever configurado com sucesso!")

    def _create_index(self, embeddings):
        """Método interno para criar o índice de um PDF"""
        print(f"Iniciando processamento do PDF: {self.pdf_path}")
        
        with LoadingIndicator("Lendo PDF") as loading:
            loader = PyPDFLoader(file_path=self.pdf_path, extract_images=True)
            documents = loader.load()
            
        print(f"\n{'=' * 50}")
        print(f"ESTATÍSTICAS DO DOCUMENTO:")
        print(f"{'=' * 50}")
        print(f"Total de páginas lidas: {len(documents)}")
        total_tokens = sum(len(doc.page_content.split()) for doc in documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"Total de tokens: {total_tokens}")
        print(f"Total de caracteres: {total_chars}")
        print(f"{'=' * 50}")
        
        # Exibir amostra de cada página para debug
        print("\nANÁLISE DE CONTEÚDO DO DOCUMENTO:")
        for i, doc in enumerate(documents):
            content_preview = doc.page_content[:150]
            print(f"\nPágina {i+1} ({len(doc.page_content)} caracteres):")
            print(f"{content_preview}...")
            if i >= 2 and len(documents) > 5:  # Limitar a exibição para documentos grandes
                print(f"... e mais {len(documents) - 3} páginas")
                break

        # Criar um text splitter com configurações aprimoradas
        print(f"\nConfigurando text splitter avançado: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Mais granularidade na separação
            keep_separator=True
        )

        with LoadingIndicator("Dividindo documento em chunks") as loading:
            docs = text_splitter.split_documents(documents)
        
        print(f"\nDocumento dividido em {len(docs)} chunks")
        print(f"Tamanho médio dos chunks: {sum(len(doc.page_content) for doc in docs) / len(docs):.1f} caracteres")
        
        # Mostrar amostra dos chunks para debug
        if len(docs) > 0:
            print("\nAMOSTRA DE CHUNKS:")
            for i, doc in enumerate(docs[:3]):  # Mostrar apenas os 3 primeiros chunks
                print(f"\nChunk {i+1}/{len(docs)} - {len(doc.page_content)} caracteres:")
                print(f"{doc.page_content[:150]}...")
            if len(docs) > 3:
                print(f"... e mais {len(docs) - 3} chunks")

        print(f"\nCriando embeddings e índice FAISS...")
        with LoadingIndicator("Criando vetores e índice") as loading:
            self.vector_store = FAISS.from_documents(docs, embeddings)
            if not os.path.exists(self.index_path):
                os.makedirs(self.index_path)
            self.vector_store.save_local(self.index_path)
        print(f"Índice FAISS criado e salvo em {self.index_path}")
        #print(f"Dimensão dos vetores: {self.vector_store.index.d}")
        #print(f"Número de vetores no índice: {self.vector_store.index.ntotal}")

    def decompose_complex_query(self, query):
        """Decompõe uma consulta complexa em consultas mais simples para melhorar a recuperação"""
        # Detectar se a consulta contém várias perguntas ou tópicos
        if '?' in query and query.count('?') > 1:
            print("Detectada consulta multi-parte. Decompondo para melhorar recuperação...")
            # Dividir por ponto de interrogação para perguntas separadas
            sub_queries = [q.strip() + '?' for q in query.split('?') if q.strip()]
            return sub_queries
        
        # Detectar partes separadas por "and", "or", "e", "ou"
        potential_conjunctions = [' and ', ' or ', ' e ', ' ou ', ', ']
        for conjunction in potential_conjunctions:
            if conjunction in query.lower():
                print(f"Detectada consulta com múltiplos tópicos conectados por '{conjunction}'")
                sub_queries = [q.strip() for q in re.split(conjunction, query, flags=re.IGNORECASE) if q.strip()]
                return sub_queries
        
        # Se não há divisão clara, retornar a consulta original
        return [query]

    def get_enhanced_context(self, query):
        """Obtém um contexto aprimorado para consultas complexas"""
        sub_queries = self.decompose_complex_query(query)
        
        if len(sub_queries) > 1:
            print(f"Consulta dividida em {len(sub_queries)} partes para busca individualizada:")
            for i, sub_q in enumerate(sub_queries):
                print(f"  {i+1}. '{sub_q}'")
            
            all_docs = []
            # Para cada subconsulta, recuperar documentos relevantes
            for sub_q in sub_queries:
                print(f"\nBuscando por: '{sub_q}'")
                sub_docs = self.retriever.invoke(sub_q)
                print(f"  Encontrados {len(sub_docs)} documentos relevantes")
                all_docs.extend(sub_docs)
            
            # Remover duplicatas mantendo a ordem
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            print(f"Total de {len(unique_docs)} documentos únicos recuperados após remoção de duplicatas")
            return unique_docs
        else:
            # Para consultas simples, usar o método padrão
            print("Usando método de recuperação padrão para consulta simples")
            return self.retriever.invoke(query)

    def ask_optimized(self, question):
        """Método otimizado para consultar o documento com base em uma pergunta"""
        if question in self.response_cache:
            print("Resposta encontrada no cache!")
            return self.response_cache[question]

        print(f"\nProcessando pergunta: '{question}'")
        max_retries = 2

        for attempt in range(max_retries):
            # Cria a mensagem do indicador de carregamento
            loading_message = f"Pensando sobre sua pergunta (tentativa {attempt+1}/{max_retries})"
            loading = LoadingIndicator(loading_message)
            loading.start()
            
            try:
                # Monitorar tempo de recuperação
                retrieval_start = time.time()
                
                # Usar estratégia de recuperação aprimorada para consultas complexas
                docs = self.get_enhanced_context(question)
                # Paramos o indicador de loading antes de continuar com o output
                loading.stop()
                
                retrieval_time = time.time() - retrieval_start
                print(f"Recuperação concluída em {retrieval_time:.2f}s - {len(docs)} documentos encontrados")
                
                if not docs:
                    print("ALERTA: Nenhum documento relevante encontrado")
                    return "Não foram encontrados documentos relevantes para essa pergunta."

                # Formatar os documentos recuperados e criar contexto
                context = format_docs(docs)
                context_size = len(context)
                print(f"Contexto gerado: {context_size} caracteres")
                
                if context_size > 15000:
                    print(f"AVISO: Contexto muito grande ({context_size} caracteres).")
                
                # Monitorar tempo de geração da resposta
                generation_start = time.time()
                
                # Usar Ollama para gerar a resposta
                print("Enviando consulta para o modelo Ollama...")
                stream = ollama.chat(
                    model="llama3.2", 
                    messages=[
                        {   
                            'role': 'user',
                            'content': f'''Você é um assistente de QA especializado em responder perguntas com base em documentos. 
Sua tarefa é fornecer respostas completas e precisas, sempre citando entre aspas e formatado a fonte das informações com base no contexto fornecido.
Considere todas as partes da pergunta e certifique-se de responder a cada aspecto.
Use trechos diretos do contexto quando possível e indique claramente de onde a informação (página, seção, parágrafo) foi extraída.
Se apenas uma parte da resposta estiver disponível no contexto, forneça o que for possível encontrar e indique o que está faltando.
Se não houver informações relevantes no contexto, informe que não há dados disponíveis. 

Contexto: {context}

Pergunta: {question}

Resposta:'''
                        },
                    ],
                    stream=True 
                    #options={
                    #    "temperature": 0.1,  # Baixa temperatura para respostas mais determinísticas
                    #    "num_predict": 2048,  # Limite de tokens para prever
                    #    "top_k": 40,         # Número de tokens mais prováveis a considerar
                    #    "top_p": 0.9         # Probabilidade cumulativa para amostragem de núcleo
                    #}
                )
                
                # Processar os chunks da resposta
                answer = ""          
                # Linha completamente limpa antes de iniciar a resposta                
                
                """ loading_message = "Thinking"
                loading = LoadingIndicator(loading_message)
                loading.start() """
                
                print("\nGerando resposta:", end='', flush=True)                
                for chunk in stream:
                    content = chunk['message']['content']
                    answer += content  # Concatenar para formar a resposta completa
                    print(content, end='', flush=True)  # Imprimir cada chunk em tempo real
                
                #loading.stop()
                
                # Finalizar com informações sobre o tempo
                generation_time = time.time() - generation_start
                print(f"\n\nResposta gerada em {generation_time:.2f}s")
                
                # Salvar no cache para consultas futuras
                self.response_cache[question] = answer
                return answer

            except Exception as e:
                # Certifica-se de parar o indicador de loading antes de mostrar o erro
                loading.stop()
                print(f"ERRO: Tentativa {attempt+1} falhou com erro: {str(e)}")
                import traceback
                print(f"Detalhes do erro: {traceback.format_exc()}")
                
                if attempt == max_retries - 1:
                    return f"Erro ao processar a pergunta: {str(e)}"
            finally:
                # Garantir que o indicador de loading seja interrompido em qualquer circunstância
                if loading.is_running:
                    loading.stop()

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
    """Remover índices de PDFs não encontrados e verificar integridade dos índices"""
    if not os.path.exists(INDICES_DIR):
        return

    print("\nVerificando integridade e limpando índices não utilizados...")
    
    # Listar todos os índices existentes
    indices = [d for d in os.listdir(INDICES_DIR) if os.path.isdir(os.path.join(INDICES_DIR, d)) and d.startswith("index_")]
    print(f"Encontrados {len(indices)} índices no diretório {INDICES_DIR}")
    
    # Verificar cada índice
    for index_dir in indices:
        pdf_name = index_dir.replace("index_", "") + ".pdf"
        pdf_path = os.path.join(PDFS_DIR, pdf_name)
        index_path = os.path.join(INDICES_DIR, index_dir)
        
        # Verificar se o PDF correspondente existe
        if not os.path.exists(pdf_path):
            print(f"LIMPEZA: Removendo índice para PDF não encontrado: {pdf_name}")
            try:
                shutil.rmtree(index_path)
            except Exception as e:
                print(f"Erro ao remover índice: {e}")
            continue
        
        # Verificar integridade do índice
        try:
            # Verificar se os arquivos necessários existem
            index_faiss = os.path.join(index_path, "index.faiss")
            index_pkl = os.path.join(index_path, "index.pkl")
            
            if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
                print(f"AVISO: Índice incompleto para {pdf_name}. Será reconstruído quando usado.")
                continue
                
            # Verificar o tamanho dos arquivos
            faiss_size = os.path.getsize(index_faiss)
            pkl_size = os.path.getsize(index_pkl)
            
            if faiss_size < 1000 or pkl_size < 100:  # Tamanhos mínimos esperados
                print(f"AVISO: Índice suspeito para {pdf_name} (tamanhos: faiss={faiss_size}B, pkl={pkl_size}B)")
                print(f"       O índice será reconstruído quando o PDF for usado.")
                
        except Exception as e:
            print(f"ERRO ao verificar índice {index_dir}: {e}")
    
    print("Verificação de índices concluída.")

def verify_workspace_integrity():
    """Verifica a integridade do ambiente de trabalho"""
    print("\nVerificando integridade do ambiente de trabalho...")
    
    # Verificar diretórios principais
    for directory in [INDICES_DIR, PDFS_DIR]:
        if not os.path.exists(directory):
            print(f"Criando diretório ausente: {directory}")
            os.makedirs(directory)
    
    # Verificar arquivos PDF no diretório pdfs/
    if os.path.exists(PDFS_DIR):
        pdf_count = len([f for f in os.listdir(PDFS_DIR) if f.lower().endswith('.pdf')])
        print(f"PDFs encontrados no diretório {PDFS_DIR}: {pdf_count}")
    
    # Verificar dependências críticas
    try:
        import faiss
        print(f"Biblioteca FAISS verificada: {faiss.__version__}")
    except (ImportError, AttributeError):
        print("ALERTA: FAISS não encontrado ou versão não identificada!")

    try:
        import sentence_transformers
        print(f"Sentence Transformers verificado: {sentence_transformers.__version__}")
    except (ImportError, AttributeError):
        print("ALERTA: Sentence Transformers não encontrado ou versão não identificada!")
        
    # Verificar Ollama
    try:
        import ollama
        print("Biblioteca Ollama encontrada")
    except ImportError:
        print("ALERTA: Biblioteca Ollama não encontrada!")
        
    print("Verificação de ambiente concluída.")

if __name__ == "__main__":
    print_header()

    # Verificar integridade do ambiente
    verify_workspace_integrity()
    
    # Limpar índices não utilizados
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

            chat.ask_optimized(user_question)
    except Exception as e:
        print(f"\nErro: {e}")
        print("Tente novamente com um arquivo PDF válido.")