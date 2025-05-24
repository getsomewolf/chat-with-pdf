import pytest
from src.core.prompt_builder import PromptBuilder # Assuming prompt_builder.py is in project root

def test_prompt_builder_builds_correct_prompt():
    builder = PromptBuilder()
    context = "Document context: Page 1 contains A. Page 2 mentions B."
    question = "What is on Page 1?"
    
    prompt = builder.build(context, question)
    
    assert PromptBuilder.BASE_INSTRUCTIONS in prompt
    assert f"Contexto: {context}" in prompt
    assert f"Pergunta: {question}" in prompt
    assert prompt.endswith("Resposta:")

def test_prompt_builder_handles_empty_context_and_question():
    builder = PromptBuilder()
    prompt = builder.build("", "")
    
    assert PromptBuilder.BASE_INSTRUCTIONS in prompt
    assert "Contexto: \n\n" in prompt # Empty context
    assert "Pergunta: \n\n" in prompt # Empty question
    assert prompt.endswith("Resposta:")

# Add more tests for edge cases or different instruction sets if PromptBuilder evolves.
