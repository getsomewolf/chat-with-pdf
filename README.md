# Chat com PDF - Aplicação de Consulta a Documentos PDF (Refatorado)

Uma aplicação Python com CLI e API FastAPI que permite conversar com seus documentos PDF utilizando um modelo de linguagem local.

> **ATENÇÃO:** Este projeto está em desenvolvimento e pode conter "bugs". Fique à vontade para testar!

## Descrição

Chat com PDF é uma ferramenta que permite fazer upload de documentos PDF (via API) e fazer perguntas sobre seu conteúdo (via API ou CLI), recebendo respostas detalhadas. A aplicação utiliza um modelo de linguagem local (LLM) através do `ollama` para processar as consultas e gerar respostas contextualizadas.

## Características Principais

- **Processamento local**: Todas as operações são realizadas localmente (requer Ollama rodando).
- **API FastAPI**:
    - Endpoint para upload de PDFs (`/upload-pdf/`) com validação de tamanho.
    - Endpoint de consulta (`/ask`) com **respostas via streaming (Server-Sent Events)**.
    - Gerenciamento de configuração via variáveis de ambiente (`.env` file).
- **Interface de Linha de Comando (CLI)**: Funcionalidade mantida para interação local.
- **Arquitetura Refatorada**:
    - Uso de Princípio da Responsabilidade Única (SRP) com `IndexService` e `QueryService`.
    - Injeção de Dependência para melhor testabilidade.
    - Operações assíncronas (`async/await`) na API e no cliente LLM.
- **Organização automática**: PDFs e índices são organizados em pastas dedicadas (`pdfs/`, `indices/`).
- **Cache de Respostas**: Para otimizar consultas repetidas.
- **Testes**: Inclui estrutura para testes unitários e de integração com `pytest`.

## Pré-requisitos

- Python 3.10 ou superior
- Docker (opcional, para rodar em container)
- Ollama instalado e rodando (com um modelo como `llama3.2` baixado)
- Espaço em disco para armazenar modelos de linguagem, PDFs e índices.
- Pelo menos 8GB de RAM (16GB recomendado).

## Instalação

1.  **Clone este repositório:**
    ```bash
    git clone https://github.com/getsomewolf/chat-with-pdf.git
    cd chat-with-pdf
    ```

2.  **Crie e configure o arquivo de ambiente:**
    Copie o arquivo `.env.example` para `.env` e ajuste as configurações conforme necessário (especialmente `OLLAMA_HOST` se o Ollama não estiver rodando em `http://localhost:11434`).
    ```bash
    cp .env.example .env
    # Edite .env com suas configurações
    ```

3.  **Instale o Pipenv** (se ainda não estiver instalado):
    ```bash
    pip install pipenv
    ```

4.  **Instale as dependências** usando Pipenv:
    ```bash
    pipenv install --dev # Instala dependências de produção e desenvolvimento (para testes)
    ```
    Isso criará um ambiente virtual e instalará todas as dependências.

5.  **Instale e configure o Ollama**:
    - Visite o [site oficial do Ollama](https://ollama.com/) e siga as instruções.
    - Baixe um modelo (ex: `llama3.2`):
      ```bash
      ollama pull llama3.2
      ```
    - Certifique-se que o servidor Ollama está rodando. Por padrão, ele escuta em `http://localhost:11434`.
      ```bash
      ollama serve
      ```
      (Mantenha este terminal aberto ou execute como serviço).

## Uso

### API FastAPI

1.  **Ative o ambiente virtual:**
    ```bash
    pipenv shell
    ```

2.  **Inicie o servidor da API:**
    ```bash
    python api.py
    ```
    Ou usando Uvicorn diretamente para mais opções (ex: workers):
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
    ```
    A API estará disponível em `http://localhost:8000` (ou conforme configurado).
    A documentação interativa (Swagger UI) estará em `http://localhost:8000/docs`.

3.  **Endpoints da API:**
    - `POST /upload-pdf/`: Faça upload de um arquivo PDF.
      - `multipart/form-data` com um campo `file`.
    - `POST /ask`: Faça uma pergunta sobre um PDF processado.
      - Corpo JSON: `{"pdf_filename": "nome_do_arquivo.pdf", "question": "Sua pergunta aqui"}`
      - A resposta é um stream de Server-Sent Events (SSE).
        - `event: sources` -> `data: {"sources": ["Fonte 1", ...]}`
        - `event: text_chunk` -> `data: {"chunk": "Parte da resposta..."}`
        - `event: error` -> `data: {"error": "Mensagem de erro..."}`
        - `event: end_stream` -> `data: {"message": "Stream ended."}`

### Interface de Linha de Comando (CLI)

1.  **Ative o ambiente virtual:**
    ```bash
    pipenv shell
    ```

2.  **Execute a aplicação CLI:**
    ```bash
    python main.py
    ```

3.  Siga as instruções no console para selecionar um PDF e fazer perguntas.
    - Digite `ajuda` ou `help` para ver sugestões e comandos.
    - Digite `reindex` para forçar a reindexação do PDF atual.
    - Digite `sair`, `exit` ou `quit` para encerrar.

## Estrutura do Projeto (Principais Arquivos)

- `api.py`: Lógica da API FastAPI.
- `main.py`: Lógica da Interface de Linha de Comando (CLI).
- `services.py`: Contém `IndexService` (para indexação) e `QueryService` (para consultas).
- `llm_client.py`: Cliente assíncrono para interagir com Ollama.
- `config.py`: Carrega configurações de `.env` usando Pydantic.
- `utils/`: Módulos utilitários.
- `pdfs/`: Diretório padrão para PDFs.
- `indices/`: Diretório padrão para índices FAISS.
- `tests/`: Contém testes unitários e de integração.
- `Pipfile`, `Pipfile.lock`: Gerenciamento de dependências.
- `Dockerfile`: Para containerização da aplicação.
- `.env.example`: Arquivo de exemplo para variáveis de ambiente.

## Configuração

As configurações principais são gerenciadas através do arquivo `.env`. Veja `.env.example` para todas as opções disponíveis, incluindo:
- `PDFS_DIR`, `INDICES_DIR`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKING_MODE`
- `EMBEDDING_MODEL_NAME`, `OLLAMA_MODEL_NAME`, `OLLAMA_HOST`, `OLLAMA_TIMEOUT`
- Configurações do Retriever (K values, thresholds)
- `API_PDF_MAX_SIZE_MB`
- `UVICORN_HOST`, `UVICORN_PORT`, `UVICORN_TIMEOUT_KEEP_ALIVE`

## Desenvolvimento e Testes

1.  **Instale dependências de desenvolvimento:**
    ```bash
    pipenv install --dev
    ```
2.  **Execute os testes:**
    Ative o ambiente (`pipenv shell`) e rode:
    ```bash
    pytest
    ```
    Para ver o relatório de cobertura de código:
    ```bash
    pytest --cov=. --cov-report html
    ```
    O relatório HTML estará em `htmlcov/index.html`.

## Docker (Opcional)

1.  **Construa a imagem Docker:**
    ```bash
    docker build -t chat-with-pdf-app .
    ```

2.  **Execute o container:**
    Certifique-se que o Ollama está acessível pela rede do container (ex: usando `host.docker.internal` para `OLLAMA_HOST` no `.env` se o Ollama estiver rodando no host, ou configurando uma rede Docker comum).
    ```bash
    docker run -p 8000:8000 \
           -v ./pdfs:/app/pdfs \
           -v ./indices:/app/indices \
           -v ./.env:/app/.env \ # Mapeia o arquivo .env para dentro do container
           --add-host=host.docker.internal:host-gateway \ # Para Ollama no host (Linux: use --network="host" ou IP)
           chat-with-pdf-app
    ```
    Ajuste `OLLAMA_HOST` no seu arquivo `.env` para `http://host.docker.internal:11434` (ou o IP do host se `--network="host"` não for usado/disponível).

## Resolução de Problemas

- **Erro "Ollama não está respondendo" / "Connection refused"**:
    - Verifique se o servidor Ollama está ativo (`ollama serve`).
    - Confirme que `OLLAMA_HOST` em `.env` está correto e acessível da aplicação/container.
    - Verifique logs do Ollama.
- **Respostas lentas**:
    - Use um modelo menor ou mais rápido.
    - Aumente os recursos de hardware.
- **Índice corrompido/desatualizado**:
    - Para a CLI, use o comando `reindex`.
    - Para a API, re-uploade o PDF para forçar a reindexação.
    - Manualmente, delete o subdiretório correspondente em `indices/`.

## Contribuições

Contribuições são bem-vindas! Abra um issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença Apache-2.0 - veja o arquivo LICENSE para mais detalhes.