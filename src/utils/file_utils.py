import os
import glob
import shutil
from src.config.settings import settings # Assuming settings are now in config.py

def list_available_pdfs() -> list[str]:
    """Lists all available PDFs in the configured PDFS_DIR and current directory."""
    pdfs_in_dir = [os.path.join(settings.PDFS_DIR, f) for f in os.listdir(settings.PDFS_DIR) if f.lower().endswith('.pdf')]
    # Also check current working directory for PDFs not yet in PDFS_DIR
    pdfs_in_current = [f for f in glob.glob("*.pdf") if os.path.abspath(f) != os.path.abspath(os.path.join(settings.PDFS_DIR, os.path.basename(f)))]
    
    all_pdfs = pdfs_in_dir + pdfs_in_current
    # Return unique paths
    return sorted(list(set(all_pdfs)))


def has_index(pdf_filename_or_path: str) -> bool:
    """Checks if an index exists for the given PDF filename or path."""
    basename = os.path.basename(pdf_filename_or_path).split('.')[0]
    index_path = os.path.join(settings.INDICES_DIR, f"index_{basename}")
    return os.path.exists(index_path) and len(os.listdir(index_path)) > 0

def select_pdf_cli() -> str | None:
    """CLI prompt to select a PDF."""
    all_pdfs = list_available_pdfs()

    if not all_pdfs:
        print("\nNenhum PDF encontrado no sistema.")
        pdf_path_input = input("Digite o caminho completo para um arquivo PDF: ")
        if not pdf_path_input or not os.path.exists(pdf_path_input):
            print("Caminho inválido ou arquivo não encontrado.")
            return None
        return pdf_path_input

    print("\nPDFs disponíveis:")
    for i, pdf_path in enumerate(all_pdfs, 1):
        indexed_status = " [indexado]" if has_index(pdf_path) else ""
        print(f"{i}. {os.path.basename(pdf_path)}{indexed_status}")

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
        print("Nenhuma seleção feita.")
        return None

def cleanup_unused_indices_cli():
    """Removes indices for PDFs that no longer exist and checks index integrity."""
    if not os.path.exists(settings.INDICES_DIR):
        return

    print("\nVerificando integridade e limpando índices não utilizados...")
    
    indices = [d for d in os.listdir(settings.INDICES_DIR) if os.path.isdir(os.path.join(settings.INDICES_DIR, d)) and d.startswith("index_")]
    print(f"Encontrados {len(indices)} índices no diretório {settings.INDICES_DIR}")
    
    for index_dir_name in indices:
        pdf_basename = index_dir_name.replace("index_", "")
        pdf_filename = f"{pdf_basename}.pdf"
        pdf_path_in_pdfs_dir = os.path.join(settings.PDFS_DIR, pdf_filename)
        current_index_path = os.path.join(settings.INDICES_DIR, index_dir_name)
        
        if not os.path.exists(pdf_path_in_pdfs_dir):
            print(f"LIMPEZA: PDF '{pdf_filename}' não encontrado em '{settings.PDFS_DIR}'. Removendo índice '{index_dir_name}'.")
            try:
                shutil.rmtree(current_index_path)
            except Exception as e:
                print(f"Erro ao remover índice '{current_index_path}': {e}")
            continue
        
        try:
            index_faiss = os.path.join(current_index_path, "index.faiss")
            index_pkl = os.path.join(current_index_path, "index.pkl")
            
            if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
                print(f"AVISO: Índice incompleto para {pdf_filename} em '{current_index_path}'. Será reconstruído quando usado.")
                continue
                
            faiss_size = os.path.getsize(index_faiss)
            pkl_size = os.path.getsize(index_pkl)
            
            if faiss_size < 1000 or pkl_size < 100: # Heuristic minimum sizes
                print(f"AVISO: Índice suspeito para {pdf_filename} (tamanhos: faiss={faiss_size}B, pkl={pkl_size}B). Será reconstruído.")
                
        except Exception as e:
            print(f"ERRO ao verificar índice {index_dir_name}: {e}")
    
    print("Verificação de índices concluída.")

def ensure_pdf_is_in_pdfs_dir(pdf_path_or_filename: str) -> str:
    """
    Ensures the PDF is in the configured PDFS_DIR.
    If it's outside, it's copied. Returns the path within PDFS_DIR.
    """
    pdf_basename = os.path.basename(pdf_path_or_filename)
    target_pdf_path = os.path.join(settings.PDFS_DIR, pdf_basename)

    if not os.path.exists(pdf_path_or_filename):
        # If only filename was given, check if it's already in PDFS_DIR
        if os.path.exists(target_pdf_path):
            return target_pdf_path
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path_or_filename} nem em {target_pdf_path}")

    # If full path was given and it's not already in PDFS_DIR
    if os.path.abspath(pdf_path_or_filename) != os.path.abspath(target_pdf_path):
        if not os.path.exists(target_pdf_path) or os.path.getmtime(pdf_path_or_filename) > os.path.getmtime(target_pdf_path):
            shutil.copy2(pdf_path_or_filename, target_pdf_path)
            print(f"PDF '{pdf_basename}' copiado para '{settings.PDFS_DIR}'.")
    
    return target_pdf_path
