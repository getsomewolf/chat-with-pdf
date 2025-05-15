import os
from langchain_core.documents import Document

def format_docs_for_cli(docs: list[Document]) -> str:
    """Formats documents for CLI output, including source information and previews."""
    formatted_text = ""
    print(f"Formatando {len(docs)} documentos recuperados:")
    
    for i, doc in enumerate(docs):
        source_info = ""
        if doc.metadata and 'source' in doc.metadata:
            source_path = doc.metadata['source']
            page_info = f", Página {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
            paragraph_info = f", Parágrafo {doc.metadata.get('paragraph', 'N/A')}" if 'paragraph' in doc.metadata else ""
            chunk_info = f", Chunk {doc.metadata.get('chunk_index', 'N/A')}" if 'chunk_index' in doc.metadata else ""
            source_info = f"[Fonte: {os.path.basename(source_path)}{page_info}{paragraph_info}{chunk_info}]"
        
        chunk_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"Chunk {i+1}/{len(docs)} {source_info}:\n{chunk_preview}\n")
        
        formatted_text += f"{doc.page_content}\n"
        if source_info:
            formatted_text += f"{source_info}\n"
        formatted_text += "\n" + "-" * 40 + "\n"
    
    return formatted_text.strip()

def format_docs_for_api(docs: list[Document]) -> tuple[str, list[str]]:
    """Formats documents for API: returns context text and a list of source strings."""
    context_text = ""
    sources: list[str] = []
    for doc in docs:
        context_text += doc.page_content + "\n"
        source_str = "Fonte: "
        if doc.metadata and 'source' in doc.metadata:
            source_str += os.path.basename(doc.metadata['source'])
        else:
            source_str += "Desconhecida"
        
        if 'page' in doc.metadata:
            source_str += f", Página {doc.metadata.get('page', 'N/A')}"
        if 'paragraph' in doc.metadata:
            source_str += f", Parágrafo {doc.metadata.get('paragraph', 'N/A')}"
        if 'chunk_index' in doc.metadata:
             source_str += f", Chunk Index {doc.metadata.get('chunk_index', 'N/A')}"
        
        if source_str != "Fonte: Desconhecida": # Only add if some metadata was found
            sources.append(source_str)
            
    # Remove duplicate sources while preserving order
    unique_sources = []
    for s in sources:
        if s not in unique_sources:
            unique_sources.append(s)
            
    return context_text.strip(), unique_sources
