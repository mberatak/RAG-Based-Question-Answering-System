"""Dokuman yukleme modulu.

PDF, TXT, DOCX ve CSV formatlarindaki dokumanlari LangChain loader'lari ile yukler.
Belirtilen dizindeki tum desteklenen dosyalari otomatik olarak tarar ve yukler.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from tqdm import tqdm
from colorama import Fore, Style

from src.config import config, logger


# Desteklenen dosya uzantilari
SUPPORTED_EXTENSIONS = {
    ".pdf": "PyPDFLoader",
    ".txt": "TextLoader",
    ".docx": "Docx2txtLoader",
    ".csv": "CSVLoader",
    ".md": "TextLoader",
}


def _get_loader(file_path: str):
    """Dosya uzantisina gore uygun LangChain loader'ini dondur.

    Args:
        file_path: Yuklenecek dosyanin yolu.

    Returns:
        Uygun LangChain loader nesnesi.

    Raises:
        ValueError: Desteklenmeyen dosya formati.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Desteklenmeyen dosya formati: {ext}")


def load_documents(
    directory: Optional[str] = None,
    file_types: Optional[List[str]] = None,
) -> List[Document]:
    """Belirtilen dizindeki tum desteklenen dokumanlari yukle.

    Args:
        directory: Dokumanlarin bulundugu dizin. None ise config'ten alinir.
        file_types: Filtrelenecek dosya uzantilari (or: ['.pdf', '.txt']).
                    None ise tum desteklenen formatlar yuklenir.

    Returns:
        List[Document]: Yuklenen dokumanlarin listesi.
    """
    directory = directory or str(config.documents_dir)
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Dizin bulunamadi: {directory}")
        print(f"{Fore.YELLOW} Dizin bulunamadi: {directory}{Style.RESET_ALL}")
        return []

    # Desteklenen dosyalari bul
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        if file_types and ext not in file_types:
            continue
        files.extend(dir_path.glob(f"*{ext}"))

    if not files:
        logger.warning(f"Dizinde desteklenen dosya bulunamadi: {directory}")
        print(f"{Fore.YELLOW} Dizinde desteklenen dosya bulunamadi: {directory}{Style.RESET_ALL}")
        return []

    print(f"\n{Fore.CYAN} Dokuman Yukleme Baslatiliyor...{Style.RESET_ALL}")
    print(f"  Dizin: {directory}")
    print(f"  Bulunan dosya sayisi: {Fore.GREEN}{len(files)}{Style.RESET_ALL}")
    print()

    all_documents: List[Document] = []
    loaded_count = 0
    error_count = 0
    total_chars = 0

    for file_path in tqdm(files, desc="  Dokumanlar yukleniyor", unit="dosya"):
        try:
            loader = _get_loader(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["file_type"] = file_path.suffix.lower()
                total_chars += len(doc.page_content)

            all_documents.extend(docs)
            loaded_count += 1
            logger.info(f"Yuklendi: {file_path.name} ({len(docs)} sayfa/bolum)")

        except Exception as e:
            error_count += 1
            logger.error(f"Dosya yuklenemedi: {file_path.name} - Hata: {str(e)}")
            print(f"\n{Fore.RED} Hata: {file_path.name} - {str(e)}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}{'-' * 40}")
    print(f"  Yukleme Tamamlandi!")
    print(f"{'-' * 40}{Style.RESET_ALL}")
    print(f"  Yuklenen dosya: {Fore.GREEN}{loaded_count}{Style.RESET_ALL}")
    print(f"  Toplam dokuman/sayfa: {Fore.GREEN}{len(all_documents)}{Style.RESET_ALL}")
    print(f"  Toplam karakter: {Fore.GREEN}{total_chars:,}{Style.RESET_ALL}")
    if error_count > 0:
        print(f"  Hatali dosya: {Fore.RED}{error_count}{Style.RESET_ALL}")
    print()

    logger.info(
        f"Dokuman yukleme tamamlandi: {loaded_count} dosya, "
        f"{len(all_documents)} dokuman, {total_chars:,} karakter"
    )

    return all_documents


def load_single_document(file_path: str) -> List[Document]:
    """Tek bir dokumani yukle.

    Args:
        file_path: Yuklenecek dosyanin yolu.

    Returns:
        List[Document]: Yuklenen dokumanlarin listesi.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        ValueError: Desteklenmeyen dosya formati.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Desteklenmeyen dosya formati: {path.suffix}. "
            f"Desteklenen: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader = _get_loader(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = path.suffix.lower()

    logger.info(f"Tek dosya yuklendi: {path.name} ({len(docs)} dokuman)")
    return docs
