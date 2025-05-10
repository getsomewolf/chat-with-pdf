class PromptBuilder:
    """
    Constrói prompt para o modelo a partir de instruções base.
    """
    BASE_INSTRUCTIONS = (
        "Você é um assistente de QA especializado em responder perguntas com base em documentos.\n"
        "Sua tarefa é fornecer respostas completas e precisas, sempre citando entre aspas e formatado a fonte das informações com base no contexto fornecido.\n"
        "Considere todas as partes da pergunta e certifique-se de responder a cada aspecto.\n"
        "Use trechos diretos do contexto quando possível e indique claramente de onde a informação (página, seção, parágrafo) foi extraída.\n"
        "Se apenas uma parte da resposta estiver disponível no contexto, forneça o que for possível encontrar e indique o que está faltando.\n"
        "Se não houver informações relevantes no contexto, informe que não há dados disponíveis."
    )

    def build(self, context: str, question: str) -> str:
        return f"{self.BASE_INSTRUCTIONS}\n\nContexto: {context}\n\nPergunta: {question}\n\nResposta:"