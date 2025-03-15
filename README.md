# Chat com PDF - Aplicação de Consulta a Documentos PDF

Uma aplicação de linha de comando que permite conversar com seus documentos PDF utilizando um modelo de linguagem local.

## Descrição

Chat com PDF é uma ferramenta que permite fazer perguntas sobre o conteúdo de documentos PDF e receber respostas detalhadas, sem depender de serviços externos como OpenAI ou outros provedores de API. A aplicação utiliza um modelo de linguagem local (LLM) para processar as consultas e gerar respostas contextualizadas baseadas no conteúdo do documento.

## Características

- **Processamento local**: Todas as operações são realizadas localmente, sem necessidade de conexão à internet após o download do modelo
- **Independência de APIs externas**: Não requer chaves de API ou serviços pagos
- **Organização automática**: Os PDFs e índices são organizados em pastas dedicadas
- **Interface amigável**: Feedback visual durante o processamento através de indicadores de carregamento
- **Timeout inteligente**: Evita que o modelo fique preso em processamentos muito longos
- **Respostas detalhadas**: Configurado para fornecer informações completas e bem estruturadas

## Pré-requisitos

- Python 3.9 ou superior
- Espaço em disco para armazenar o modelo de linguagem (aproximadamente 4-7GB)
- Pelo menos 8GB de RAM (16GB recomendado)

## Instalação

1. Clone este repositório ou baixe os arquivos:

```bash
git clone https://github.com/seu-usuario/chat-with-pdf.git
cd chat-with-pdf
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Baixe um modelo de linguagem GGML/GGUF. Recomendamos:
   - [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) 
   - [Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

4. Coloque o arquivo do modelo na pasta `models` do projeto (será criada automaticamente na primeira execução).
   - Você pode renomear o modelo para `ggml-model.bin` ou deixar o nome original

## Uso

1. Execute o aplicativo:

```bash
python main.py
```

2. Selecione um PDF para processar:
   - O programa mostrará uma lista de PDFs disponíveis
   - PDFs que já foram processados anteriormente serão marcados como [indexado]
   - Você pode digitar o número correspondente ao PDF ou o caminho completo para um novo arquivo

3. Faça perguntas sobre o documento:
   - Digite sua pergunta e pressione Enter
   - O programa processará a pergunta e mostrará a resposta
   - Digite 'ajuda' ou 'help' para ver sugestões de perguntas
   - Digite 'sair', 'exit' ou 'quit' para encerrar o programa

## Estrutura do Projeto

- `main.py` - Arquivo principal da aplicação
- `models/` - Diretório onde os modelos de linguagem devem ser colocados
- `pdfs/` - Diretório onde os PDFs serão armazenados
- `indices/` - Diretório para os índices de vetores dos documentos processados
- `requirements.txt` - Lista de dependências do projeto

## Configuração Avançada

Você pode ajustar os seguintes parâmetros no código fonte para personalizar o comportamento:

- **Tamanho dos chunks**: Modificando o parâmetro `chunk_size` (padrão: 400 caracteres)
- **Sobreposição dos chunks**: Ajustando o parâmetro `chunk_overlap` (padrão: 100 caracteres)
- **Número de documentos recuperados**: Alterando o parâmetro `k` na pesquisa (padrão: 4)
- **Timeout**: Modificando o valor em `timeout_duration` (padrão: 120 segundos)
- **Parâmetros do modelo LLM**: Ajustando valores como `temperature`, `max_tokens`, `n_ctx`, etc.

## Resolução de Problemas

1. **Erro "Nenhum modelo LLM encontrado"**:
   - Certifique-se de ter baixado um modelo GGML/GGUF
   - Verifique se o modelo está na pasta `models/`

2. **Respostas muito lentas**:
   - Considere usar um modelo menor ou quantizado (como modelos com Q4_K_M no nome)
   - Aumente o parâmetro `n_batch` para acelerar a inferência em GPUs

3. **Erros de memória**:
   - Reduza o valor de `n_ctx` para diminuir o consumo de memória
   - Use um modelo menor ou mais quantizado

4. **Índice corrompido**:
   - Delete a pasta do índice correspondente em `indices/` para recriá-lo

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir um issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.

## Agradecimentos

- [LangChain](https://github.com/hwchase17/langchain) - Framework para aplicações de LLM
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) - Implementação eficiente de modelos de linguagem
- [FAISS](https://github.com/facebookresearch/faiss) - Biblioteca para busca de similaridade eficiente
- [TheBloke](https://huggingface.co/TheBloke) - Por disponibilizar modelos GGML/GGUF otimizados
