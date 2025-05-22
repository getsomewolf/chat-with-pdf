# Chat com PDF - Aplicação de Consulta a Documentos PDF (v0.2.0)

Uma aplicação Python que permite interagir com documentos PDF utilizando um modelo de linguagem local (LLM) via Ollama, agora com gerenciamento simplificado via Docker Compose e recuperação de contexto aprimorada.

> **Nota:** Este projeto está em desenvolvimento e pode conter "bugs". Testes são bem-vindos!
> **Novidades na v0.2.0:** Melhorias na extração de texto de PDFs, otimização da recuperação de contexto, gerenciamento de memória para índices, e integração completa com Ollama via Docker Compose.

## Descrição

Chat com PDF é uma ferramenta que permite fazer upload de documentos PDF e realizar perguntas sobre seu conteúdo, recebendo respostas detalhadas. A aplicação utiliza um modelo de linguagem local através do `ollama` para processar consultas e gerar respostas contextualizadas.

## Funcionalidades

- **Processamento Local:** Todas as operações são realizadas localmente (requer Ollama rodando, facilitado pelo Docker Compose).
- **API FastAPI:**
  - Upload de PDFs (`/upload-pdf/`).
  - Consultas com respostas via streaming (`/ask`).
- **Interface CLI:** Interação local com PDFs.
- **Arquitetura Modular:**
  - Uso de princípios de design como SRP e injeção de dependência.
  - Operações assíncronas para maior desempenho.
- **Organização Automática:** PDFs e índices organizados em pastas dedicadas.
- **Cache de Respostas e Índices:** Otimização para consultas repetidas e gerenciamento de memória para índices vetoriais.
- **Testes Automatizados:** Estrutura para testes unitários e de integração com `pytest`.
- **Docker Compose:** Configuração simplificada para rodar a aplicação e o Ollama.

## Requisitos

- Python 3.10+ (para desenvolvimento local fora do Docker)
- Docker e Docker Compose
- Pelo menos 8GB de RAM (16GB recomendado, especialmente para Ollama)

## Instalação e Execução com Docker Compose

Esta é a forma recomendada para rodar a aplicação.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/getsomewolf/chat-with-pdf.git
    cd chat-with-pdf
    ```

2.  **Configure o ambiente:**
    Copie o arquivo de exemplo `.env.example` para `.env` e edite-o conforme necessário.
    ```bash
    cp .env.example .env
    ```
    Certifique-se de que `OLLAMA_MODEL_NAME` em `.env` (ex: `llama3.2`) corresponda a um modelo que você deseja usar.

3.  **Construa e inicie os serviços com Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    Isso irá construir a imagem da aplicação (`app`) e baixar/iniciar a imagem do Ollama (`ollama`).

4.  **Baixe um modelo LLM via Ollama (se ainda não o fez):**
    Após os contêineres estarem rodando, você pode precisar puxar o modelo LLM especificado em seu `.env` para o serviço Ollama. Abra outro terminal e execute:
    ```bash
    docker-compose exec ollama ollama pull seu_modelo_aqui 
    # Exemplo: docker-compose exec ollama ollama pull llama3.2 
    ```
    Substitua `seu_modelo_aqui` pelo nome do modelo configurado em `OLLAMA_MODEL_NAME` no arquivo `.env`. Os modelos baixados serão persistidos no volume `ollama_data`.

5.  **Acesse a API:**
    A API estará disponível em `http://localhost:PORTA`, onde `PORTA` é o valor de `UVICORN_PORT` definido no seu `.env` (padrão `8000`).
    - Documentação interativa (Swagger UI): `http://localhost:PORTA/docs`

## Uso

### API FastAPI (via Docker Compose)

Após iniciar os serviços com `docker-compose up`, utilize um cliente HTTP (como Postman, curl, ou a interface Swagger UI em `http://localhost:PORTA/docs`) para interagir com os endpoints:

-   **`POST /upload-pdf/`**: Faça upload de um arquivo PDF.
    -   Corpo: `multipart/form-data` com uma chave `file` contendo o arquivo PDF.
-   **`POST /ask`**: Faça uma pergunta sobre um PDF previamente carregado.
    -   Corpo: JSON com `pdf_filename` (nome do arquivo PDF, ex: "documento.pdf") e `question`.
    -   A resposta é transmitida via Server-Sent Events (SSE).

### CLI (Interface de Linha de Comando)

Você pode executar a CLI dentro do contêiner da aplicação Docker:

1.  Certifique-se de que os contêineres estão rodando: `docker-compose up -d`
2.  Execute o script CLI:
    ```bash
    docker-compose exec app python src/cli/main.py
    ```
3.  Siga as instruções no console para selecionar um PDF e fazer perguntas.

### Desenvolvimento Local (sem Docker Compose para a app, mas Ollama pode ser Docker ou local)

1.  Clone o repositório e configure o `.env` como descrito acima.
2.  Instale as dependências:
    ```bash
    pip install pipenv
    pipenv install --dev
    ```
3.  Ative o ambiente virtual:
    ```bash
    pipenv shell
    ```
4.  **Configure o Ollama:**
    -   Certifique-se de que o Ollama está instalado e rodando. Você pode usar o serviço Ollama do Docker Compose (`docker-compose up -d ollama`) ou uma instalação local do Ollama.
    -   Se usando uma instalação local do Ollama, verifique se `OLLAMA_HOST` no `.env` aponta para ele (ex: `http://localhost:11434`).
    -   Baixe um modelo (ex.: `llama3.2`):
        ```bash
        # Se Ollama está rodando localmente:
        ollama pull llama3.2
        # Se Ollama está rodando via Docker Compose (do passo anterior):
        # docker-compose exec ollama ollama pull llama3.2
        ```
5.  **Inicie a API FastAPI (localmente):**
    ```bash
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
6.  **Execute a CLI (localmente):**
    ```bash
    python src/cli/main.py
    ```

## Estrutura do Projeto

- `src/`: Contém todo o código fonte da aplicação.
  - `api/main.py`: Lógica da API FastAPI.
  - `cli/main.py`: Interface CLI.
  - `core/`: Lógica de negócios principal (serviços, LLM client, prompt builder, event manager).
  - `infra/`: Componentes de infraestrutura (repositórios de PDF/vetor, factories, estratégias de chunking/retrieval).
  - `config/`: Configurações do projeto (`settings.py`).
  - `utils/`: Módulos utilitários.
- `pdfs/`: Diretório para PDFs (montado como volume no Docker).
- `indices/`: Diretório para índices FAISS (montado como volume no Docker).
- `tests/`: Testes unitários e de integração.
- `Dockerfile`: Define a imagem Docker para a aplicação.
- `docker-compose.yml`: Orquestra os serviços da aplicação e do Ollama.
- `.env.example`: Arquivo de exemplo para variáveis de ambiente.
- `Pipfile` & `Pipfile.lock`: Gerenciamento de dependências Python com Pipenv.

## Testes

1.  Ative o ambiente virtual (se não estiver usando Docker para testes):
    ```bash
    pipenv shell
    ```
2.  Execute os testes:
    ```bash
    pytest
    ```
3.  Gere o relatório de cobertura:
    ```bash
    pytest --cov=src --cov-report html
    ```
    O relatório estará disponível em `htmlcov/index.html`.

## Contribuições

Contribuições são bem-vindas! Abra um issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença Apache-2.0. Consulte o arquivo LICENSE para mais detalhes.