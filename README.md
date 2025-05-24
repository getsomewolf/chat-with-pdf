# Alfred Agent Chat com PDF - Aplicação de Consulta a Documentos PDF (v0.2.0)

Uma aplicação Python que permite interagir com documentos PDF utilizando um modelo de linguagem local (LLM) via Ollama, com gerenciamento simplificado via Docker Compose e recuperação de contexto aprimorada com estratégias híbridas de busca.

> **Nota:** Este projeto está em desenvolvimento ativo e pode conter "bugs". Testes e contribuições são bem-vindos!
> 
> **Novidades na v0.2.0:** Melhorias na extração de texto com PyMuPDFLoader, estratégias híbridas de recuperação (Vector + BM25), gerenciamento de memória para índices, integração completa com Ollama via Docker Compose, e suporte a GPU configurável.

## Descrição

Chat com PDF é uma ferramenta que permite fazer upload de documentos PDF e realizar perguntas sobre seu conteúdo, recebendo respostas detalhadas e contextualizadas. A aplicação utiliza:

- **Processamento local** com Ollama para LLM
- **Embeddings** com HuggingFace (sentence-transformers)
- **Busca híbrida** combinando similaridade vetorial (FAISS) e BM25
- **Streaming de respostas** via Server-Sent Events (SSE)
- **Cache inteligente** para otimização de performance

## Funcionalidades

- **Processamento 100% Local:** Todas as operações são realizadas localmente via Ollama
- **API FastAPI:**
  - Upload de PDFs (`/upload-pdf/`)
  - Consultas com respostas via streaming (`/ask`)
  - Documentação interativa (Swagger UI)
- **Interface CLI:** Interação direta via linha de comando
- **Arquitetura Modular:**
  - Uso de princípios SOLID (SRP, injeção de dependência)
  - Operações assíncronas para performance
  - Sistema de eventos para observabilidade
- **Estratégias de Chunking:** Tokens, parágrafos ou híbrida
- **Recuperação Híbrida:** Combina busca vetorial e BM25 para maior precisão
- **Cache Multinível:** Respostas, índices e serviços
- **Suporte a GPU:** Configurável para CUDA, MPS, NPU ou CPU
- **Testes Automatizados:** Estrutura completa com pytest

## Requisitos

- **Docker e Docker Compose** (recomendado)
- **Python 3.10+** (para desenvolvimento local)
- **Pelo menos 8GB de RAM** (16GB recomendado para modelos maiores)
- **GPU opcional:** NVIDIA CUDA, Apple MPS ou NPU para aceleração

## Configuração e Instalação

### 1. Clone o Repositório

```bash
git clone <seu-repositorio>
cd chat-with-pdf
```

### 2. Configure as Variáveis de Ambiente

Copie e configure o arquivo de ambiente:

```bash
cp .env.example .env
```

**Principais configurações no `.env`:**

```bash
# Modelo Ollama (certifique-se que existe)
OLLAMA_MODEL_NAME="llama3.2"

# Configuração de dispositivo para embeddings/computação
DEVICE_CONFIGURATION="cpu"  # Opções: "cpu", "cuda", "mps", "npu"

# Configurações de chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNKING_MODE="both"  # "tokens", "paragraphs", "both"

# Configurações de recuperação híbrida
INITIAL_VECTOR_K=50          # Documentos iniciais da busca vetorial
VECTOR_DISTANCE_THRESHOLD=1.0 # Limite de distância L2
FINAL_BM25_K=6              # Documentos finais do BM25

# Porta da API
UVICORN_PORT=8000
```

## Execução com Docker Compose (Recomendado)

### Início Rápido

```bash
# 1. Construir e iniciar todos os serviços
docker-compose up --build -d

# 2. Verificar se os serviços estão rodando
docker-compose ps

# 3. Baixar o modelo LLM (substituir por seu modelo)
docker-compose exec ollama ollama pull llama3.2

# 4. Acessar a API
# http://localhost:8000/docs (Swagger UI)
```

### Estrutura dos Serviços

O `docker-compose.yml` define dois serviços principais:

- **`app`**: Aplicação FastAPI principal
- **`ollama`**: Servidor Ollama para LLM

**Volumes persistentes:**
- `./pdfs:/app/pdfs` - Armazenamento de PDFs
- `./indices:/app/indices` - Índices FAISS
- `${USERPROFILE}/.ollama/models:/root/.ollama/models` - Modelos Ollama

## Uso da Aplicação

### API FastAPI

**Upload de PDF:**
```bash
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@documento.pdf"
```

**Fazer pergunta (streaming):**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"pdf_filename": "documento.pdf", "question": "Qual é o tema principal?"}'
```

### Interface CLI

```bash
# Executar CLI dentro do container
docker-compose exec app python -m src.cli.main

# Ou com desenvolvimento local
python -m src.cli.main
```

## Depuração e Monitoramento

### Logs dos Serviços

**Ver logs da aplicação:**
```bash
# Logs em tempo real
docker-compose logs app -f

# Logs das últimas 100 linhas
docker-compose logs app --tail=100

# Filtrar logs por nível (ERROR, WARNING, INFO)
docker-compose logs app | grep ERROR
```

**Ver logs do Ollama:**
```bash
# Logs em tempo real do serviço Ollama
docker-compose logs ollama -f

# Verificar inicialização do modelo
docker-compose logs ollama --tail=50

# Verificar se o modelo foi baixado com sucesso
docker-compose logs ollama | grep "pulled successfully"
```

**Logs de todos os serviços:**
```bash
# Todos os logs em tempo real
docker-compose logs -f

# Logs específicos por timestamp
docker-compose logs --since="2024-01-01T10:00:00"

# Salvar logs em arquivo para análise
docker-compose logs > debug_logs.txt
```

### Comandos de Diagnóstico

**Verificar status dos serviços:**
```bash
docker-compose ps

# Ver uso de recursos
docker stats

# Verificar se os containers estão saudáveis
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

**Acessar shell do container:**
```bash
# Container da aplicação
docker-compose exec app bash

# Container do Ollama
docker-compose exec ollama bash

# Executar comandos específicos sem shell interativo
docker-compose exec app python --version
docker-compose exec ollama ollama --version
```

**Verificar modelos disponíveis no Ollama:**
```bash
# Listar modelos instalados
docker-compose exec ollama ollama list

# Verificar tamanho dos modelos
docker-compose exec ollama du -sh /root/.ollama/models/*

# Buscar modelos disponíveis online
docker-compose exec ollama ollama search llama
```

**Testar conectividade e APIs:**
```bash
# Testar conectividade com Ollama
docker-compose exec app curl http://ollama:11434/api/version

# Testar API da aplicação
curl http://localhost:8000/docs

# Verificar endpoints disponíveis
curl http://localhost:8000/openapi.json | jq .
```

### Resolução de Problemas Específicos

**Problema: Script de inicialização falha com "curl: not found"**

O script `init-ollama.sh` foi atualizado para instalar automaticamente `curl` ou usar `wget` como fallback:

```bash
# Verificar logs de inicialização do Ollama
docker-compose logs ollama --tail=20

# Se ainda houver problemas, rebuildar o container
docker-compose down
docker-compose up --build ollama
```

**Problema: Modelo não baixa automaticamente**

```bash
# Verificar se o modelo especificado existe
docker-compose exec ollama ollama search llama3.2

# Baixar manualmente se necessário
docker-compose exec ollama ollama pull llama3.2

# Verificar espaço em disco
docker-compose exec ollama df -h
```

**Problema: Timeout durante download do modelo**

```bash
# O script tem retry automático, mas você pode monitorar:
docker-compose logs ollama -f

# Para modelos grandes, considere baixar antes de subir os containers:
# 1. Subir apenas Ollama
docker-compose up -d ollama

# 2. Baixar modelo manualmente
docker-compose exec ollama ollama pull llama3.2

# 3. Subir o resto
docker-compose up -d app
```

## Configurações Avançadas do Dockerfile

### Variáveis de Ambiente Comentadas

No `Dockerfile`, há algumas variáveis de ambiente importantes comentadas:

```dockerfile
# Set environment variables for HuggingFace tokenizers and Python
# ENV TOKENIZERS_PARALLELISM="false"
# ENV PYTHONUNBUFFERED=1
# REMOVE the following line if you want GPU access:
# ENV CUDA_VISIBLE_DEVICES=""
```

#### Explicação das Variáveis:

**1. `TOKENIZERS_PARALLELISM="false"`**
- **Propósito:** Desabilita paralelização dos tokenizers do HuggingFace
- **Impacto na eficiência:** 
  - ✅ **Vantagem:** Evita deadlocks em ambientes containerizados
  - ❌ **Desvantagem:** Tokenização mais lenta em textos muito grandes
  - 🎯 **Recomendação:** Manter desabilitado em produção para estabilidade

**2. `PYTHONUNBUFFERED=1`**
- **Propósito:** Força output imediato do Python (sem buffer)
- **Impacto na execução:**
  - ✅ **Vantagem:** Logs aparecem imediatamente no Docker
  - ✅ **Melhor depuração:** Output em tempo real
  - ❌ **Overhead mínimo:** Ligeiramente menos eficiente para I/O intensivo

**3. `CUDA_VISIBLE_DEVICES=""`**
- **Propósito:** Oculta GPUs do container quando definido como vazio
- **Impacto na inferência:**
  - ⚠️ **CPU forçado:** Mesmo com GPU disponível, usará apenas CPU
  - 🐌 **Performance:** Embeddings e inferência significativamente mais lentos
  - 💾 **Memória:** Menor uso de VRAM, mais uso de RAM
  - 🔧 **Compatibilidade:** Evita erros CUDA em ambientes sem suporte adequado

#### Recomendações de Configuração:

**Para ambientes com GPU:**
```dockerfile
ENV TOKENIZERS_PARALLELISM="false"
ENV PYTHONUNBUFFERED=1
# Remover ou comentar a linha abaixo para usar GPU:
# ENV CUDA_VISIBLE_DEVICES=""
```

**Para ambientes apenas CPU:**
```dockerfile
ENV TOKENIZERS_PARALLELISM="false"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""  # Força uso de CPU
```

### Configuração de GPU no Docker Compose

Para usar GPU, descomente as seções relevantes no `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # ou 'all' para todas as GPUs
          capabilities: [gpu]
```

## Desenvolvimento Local

### Configuração sem Docker

1. **Instalar dependências:**
   ```bash
   pip install pipenv
   pipenv install --dev
   pipenv shell
   ```

2. **Configurar Ollama:**
   ```bash
   # Instalar Ollama localmente ou usar Docker apenas para Ollama
   docker-compose up -d ollama
   
   # Ou instalar Ollama nativamente e ajustar OLLAMA_HOST no .env
   # OLLAMA_HOST="http://localhost:11434"
   ```

3. **Executar aplicação:**
   ```bash
   # API
   uvicorn src.api.main:app
   
   # CLI
   python -m src.cli.main
   ```

### Estrutura do Projeto

```
src/
├── api/           # FastAPI endpoints
├── cli/           # Interface linha de comando  
├── core/          # Lógica de negócios
│   ├── services.py       # IndexService, QueryService
│   ├── llm_client.py     # Cliente Ollama
│   ├── prompt_builder.py # Construção de prompts
│   ├── event_manager.py  # Sistema de eventos
│   └── observers.py      # Observadores de eventos (ex: Logging)
├── infra/         # Infraestrutura
│   ├── pdf_repository.py      # Carregamento PDFs
│   ├── vector_store_repository.py # Gerenciamento FAISS
│   ├── embeddings_factory.py     # Factory embeddings
│   ├── chunk_strategies.py       # Estratégias chunking
│   └── retriever_strategies.py   # Estratégias recuperação
├── config/        # Configurações
└── utils/         # Utilitários
```

## Testes

### Executar Testes

```bash
# Com Docker
docker-compose exec app pytest

# Desenvolvimento local
pipenv shell
pytest

# Com cobertura
pytest --cov=src --cov-report html
```

### Estrutura de Testes

- `tests/unit/` - Testes unitários
- `tests/integration/` - Testes de integração API
- `tests/fixtures/` - Fixtures compartilhadas

## Solução de Problemas

### Problemas Comuns

**1. Ollama não responde:**
```bash
# Verificar se está rodando
docker-compose exec ollama curl http://localhost:11434/api/version

# Restart do serviço
docker-compose restart ollama
```

**2. Modelo não encontrado:**
```bash
# Listar modelos
docker-compose exec ollama ollama list

# Baixar modelo
docker-compose exec ollama ollama pull llama3.2
```

**3. Erro de memória:**
- Aumentar RAM disponível para Docker
- Usar modelo menor (ex: `llama3.2:1b`)
- Ajustar `CHUNK_SIZE` e cache settings

**4. Performance lenta:**
- Verificar `DEVICE_CONFIGURATION` no `.env`
- Considerar usar GPU se disponível
- Ajustar parâmetros de recuperação (`INITIAL_VECTOR_K`, `FINAL_BM25_K`)

## Contribuições

Contribuições são muito bem-vindas! 

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commits seguindo convenções
4. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença Apache-2.0. Consulte o arquivo `LICENSE` para mais detalhes.