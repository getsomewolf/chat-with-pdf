# Chat com PDF - Aplicação de Consulta a Documentos PDF

Uma aplicação Python que permite interagir com documentos PDF utilizando um modelo de linguagem local (LLM).

> **Nota:** Este projeto está em desenvolvimento e pode conter "bugs". Testes são bem-vindos!

## Descrição

Chat com PDF é uma ferramenta que permite fazer upload de documentos PDF e realizar perguntas sobre seu conteúdo, recebendo respostas detalhadas. A aplicação utiliza um modelo de linguagem local através do `ollama` para processar consultas e gerar respostas contextualizadas.

## Funcionalidades

- **Processamento Local:** Todas as operações são realizadas localmente (requer Ollama rodando).
- **API FastAPI:**
  - Upload de PDFs (`/upload-pdf/`).
  - Consultas com respostas via streaming (`/ask`).
- **Interface CLI:** Interação local com PDFs.
- **Arquitetura Modular:**
  - Uso de princípios de design como SRP e injeção de dependência.
  - Operações assíncronas para maior desempenho.
- **Organização Automática:** PDFs e índices organizados em pastas dedicadas.
- **Cache de Respostas:** Otimização para consultas repetidas.
- **Testes Automatizados:** Estrutura para testes unitários e de integração com `pytest`.

## Requisitos

- Python 3.10+
- Docker (opcional)
- Ollama instalado e rodando
- Pelo menos 8GB de RAM (16GB recomendado)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/getsomewolf/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. Configure o ambiente:
   ```bash
   cp .env.example .env
   # Edite o arquivo .env conforme necessário
   ```

3. Instale as dependências:
   ```bash
   pip install pipenv
   pipenv install --dev
   ```

4. Configure o Ollama:
   - Instale o Ollama seguindo as instruções do [site oficial](https://ollama.com/).
   - Baixe um modelo (ex.: `llama3.2`):
     ```bash
     ollama pull llama3.2
     ```
   - Inicie o servidor Ollama:
     ```bash
     ollama serve
     ```

## Uso

### API FastAPI

1. Ative o ambiente virtual:
   ```bash
   pipenv shell
   ```

2. Inicie o servidor:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. Acesse a API em `http://localhost:8000`.
   - Documentação interativa: `http://localhost:8000/docs`

### CLI

1. Ative o ambiente virtual:
   ```bash
   pipenv shell
   ```

2. Execute a aplicação CLI:
   ```bash
   python cli.py
   ```

3. Siga as instruções no console para selecionar um PDF e fazer perguntas.

## Estrutura do Projeto

- `api.py`: Lógica da API FastAPI.
- `cli.py`: Interface CLI.
- `services.py`: Serviços de indexação e consulta.
- `llm_client.py`: Cliente para interação com o Ollama.
- `config.py`: Configurações do projeto.
- `utils/`: Módulos utilitários.
- `pdfs/`: Diretório para PDFs.
- `indices/`: Diretório para índices FAISS.
- `tests/`: Testes unitários e de integração.

## Testes

1. Execute os testes:
   ```bash
   pytest
   ```

2. Gere o relatório de cobertura:
   ```bash
   pytest --cov=. --cov-report html
   ```
   O relatório estará disponível em `htmlcov/index.html`.

## Docker

1. Construa a imagem Docker:
   ```bash
   docker build -t chat-with-pdf-app .
   ```

2. Execute o container:
   ```bash
   docker run -p 8000:8000 \
          -v ./pdfs:/app/pdfs \
          -v ./indices:/app/indices \
          -v ./.env:/app/.env \
          chat-with-pdf-app
   ```

## Contribuições

Contribuições são bem-vindas! Abra um issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença Apache-2.0. Consulte o arquivo LICENSE para mais detalhes.