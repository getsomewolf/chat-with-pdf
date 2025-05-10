# Chat com PDF - Aplicação de Consulta a Documentos PDF

Uma aplicação de linha de comando que permite conversar com seus documentos PDF utilizando um modelo de linguagem local.

> **ATENÇÃO:** Este projeto está em desenvolvimento e pode conter "bugs". Fique à vontade para testar!

## Descrição

Chat com PDF é uma ferramenta que permite fazer perguntas sobre o conteúdo de documentos PDF e receber respostas detalhadas, sem depender de serviços externos como OpenAI ou outros provedores de API. A aplicação utiliza um modelo de linguagem local (LLM) através do `ollama` para processar as consultas e gerar respostas contextualizadas baseadas no conteúdo do documento.

## Características

- **Processamento local**: Todas as operações são realizadas localmente, sem necessidade de conexão à internet após o download do modelo.
- **Independência de APIs externas**: Não requer chaves de API ou serviços pagos.
- **Organização automática**: Os PDFs e índices são organizados em pastas dedicadas.
- **Interface amigável**: Feedback visual durante o processamento através de indicadores de carregamento.
- **Timeout inteligente**: Evita que o modelo fique preso em processamentos muito longos.
- **Respostas detalhadas**: Configurado para fornecer informações completas e bem estruturadas.
- **Decomposição de consultas complexas**: Divide perguntas complexas para melhorar a recuperação de informações.
- **Diversidade de respostas**: Utiliza Maximum Marginal Relevance (MMR) para oferecer respostas mais abrangentes.

## Pré-requisitos

- Python 3.9 ou superior
- Espaço em disco para armazenar o modelo de linguagem (aproximadamente 4-7GB)
- Pelo menos 8GB de RAM (16GB recomendado)

## Instalação

Siga os passos abaixo para configurar o ambiente e instalar as dependências necessárias:

1. **Clone este repositório ou baixe os arquivos:**

   ```bash
   git clone https://github.com/getsomewolf/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. **Instale o Pipenv**, se ainda não estiver instalado:

   ```bash
   pip install pipenv
   ```

3. **Instale as dependências** usando o Pipenv:

   ```bash
   pipenv install
   ```

   Isso criará um ambiente virtual e instalará todas as dependências listadas no Pipfile, incluindo o `ollama`.

4. **Instale o Ollama**:

   - Visite o [site oficial do Ollama](https://ollama.com/) e siga as instruções para instalar o Ollama no seu sistema operacional.
   - Após a instalação, baixe um modelo de linguagem. Recomendamos o modelo `llama3.2`, mas você pode escolher outro modelo compatível:

     ```bash
     ollama pull llama3.2
     ```

5. **Certifique-se de que o Ollama está rodando**:

   - Inicie o servidor do Ollama em um terminal separado:

     ```bash
     ollama serve
     ```

   - Mantenha este terminal aberto enquanto usa o Chat com PDF.

## Uso

1. **Ative o ambiente virtual do Pipenv:**

   ```bash
   pipenv shell
   ```

2. **Execute o aplicativo:**

   ```bash
   python main.py
   ```

3. **Selecione um PDF para processar:**
   - O programa exibirá uma lista de PDFs disponíveis no diretório `pdfs/`.
   - PDFs já processados anteriormente serão marcados como `[indexado]`.
   - Digite o número correspondente ao PDF ou o caminho completo para um novo arquivo.

4. **Faça perguntas sobre o documento:**
   - Digite sua pergunta e pressione Enter.
   - O programa processará a pergunta usando o modelo configurado no `ollama` e exibirá a resposta.
   - Digite `ajuda` ou `help` para ver sugestões de perguntas.
   - Digite `sair`, `exit` ou `quit` para encerrar o programa.

## Estrutura do Projeto

- `main.py` - Arquivo principal da aplicação.
- `pdfs/` - Diretório onde os PDFs serão armazenados.
- `indices/` - Diretório para os índices de vetores dos documentos processados.
- `Pipfile` - Arquivo que define as dependências do projeto, agora incluindo `ollama`.
- `Pipfile.lock` - Arquivo gerado automaticamente para bloquear as versões das dependências.

## Configuração Avançada

Você pode personalizar o comportamento ajustando os seguintes parâmetros no código fonte (`main.py`):

- **Tamanho dos chunks**: Modifique `chunk_size` (padrão: 1000 caracteres).
- **Sobreposição dos chunks**: Ajuste `chunk_overlap` (padrão: 200 caracteres).
- **Recuperação de documentos**: O parâmetro `retrieval_k` define quantos documentos serão recuperados (padrão: 3).
- **Diversidade de resultados**: Ajuste `diversity_lambda` para balancear relevância e diversidade (padrão: 0.25).
- **Modelo de embeddings**: Utiliza "sentence-transformers/all-mpnet-base-v2" para melhor captura semântica.
- **Parâmetros do Ollama**: Ajustáveis na chamada `ollama.chat()`:
  - `temperature`: 0.1 (baixa temperatura para respostas mais determinísticas)
  - `num_predict`: 2048 (limite de tokens para prever)
  - `top_k`: 40 (número de tokens mais prováveis a considerar)
  - `top_p`: 0.9 (probabilidade cumulativa para amostragem de núcleo)

## Resolução de Problemas

1. **Erro "Ollama não está respondendo":**
   - Verifique se o servidor do Ollama está ativo (`ollama serve`).
   - Confirme que o modelo foi baixado corretamente com `ollama list`.

2. **Respostas muito lentas:**
   - Use um modelo menor ou mais rápido (ex.: `llama3.2` já é otimizado).
   - Aumente os recursos de hardware, se possível.

3. **Erros de memória:**
   - Reduza o número de documentos recuperados (`retrieval_k`) ou use um modelo menor.

4. **Índice corrompido:**
   - Delete o diretório correspondente em `indices/` para recriá-lo.
   - Alternativamente, defina `force_reindex=True` na classe ChatWithPDF.

## Contribuições

Contribuições são bem-vindas! Abra um issue ou envie um pull request para colaborar.

## Licença

Este projeto está licenciado sob a licença Apache-2.0 - veja o arquivo LICENSE para mais detalhes.

## Agradecimentos

- [LangChain](https://github.com/hwchase17/langchain) - Framework para aplicações de LLM.
- [Ollama](https://ollama.com/) - Ferramenta para rodar modelos de linguagem localmente.
- [FAISS](https://github.com/facebookresearch/faiss) - Biblioteca para busca de similaridade eficiente.
- [Hugging Face](https://huggingface.co/) - Plataforma para modelos de linguagem e embeddings.