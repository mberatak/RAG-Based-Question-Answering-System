"""Metin bolme modulu.

Yuklenen dokumanlari yapilandirilabilir boyutlarda parcalara (chunk) boler.
RecursiveCharacterTextSplitter kullanarak anlamli bolme noktalari olusturur.
"""

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from colorama import Fore, Style

from src.config import config, logger


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    custom_separators: Optional[List[str]] = None,
) -> List[Document]:
    """Dokumanlari yapilandirilabilir boyutlarda parcalara bol.

    Args:
        documents: Bolunecek dokuman listesi.
        chunk_size: Parca boyutu (karakter). None ise config'ten alinir.
        chunk_overlap: Parcalar arasi ortusme. None ise config'ten alinir.
        custom_separators: Ozel ayrac listesi.

    Returns:
        List[Document]: Bolunmus dokuman parcalarinin listesi.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    separators = custom_separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

    print(f"\n{Fore.CYAN} Metin Bolme Baslatiliyor...{Style.RESET_ALL}")
    print(f"  Chunk boyutu: {chunk_size}")
    print(f"  Chunk ortusme: {chunk_overlap}")
    print(f"  Dokuman sayisi: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # Her chunk'a ek metadata ekle
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

        if "source" in chunk.metadata:
            source = chunk.metadata["source"]
            chunk.metadata["source_short"] = source.replace("\\", "/").split("/")[-1]

    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0

    print(f"\n{Fore.GREEN}{'─' * 40}")
    print(f"  Metin Bolme Tamamlandi!")
    print(f"{'─' * 40}{Style.RESET_ALL}")
    print(f"  Toplam parca sayisi: {Fore.GREEN}{len(chunks)}{Style.RESET_ALL}")
    print(f"  Ortalama parca boyutu: {Fore.GREEN}{avg_chunk_size:.0f}{Style.RESET_ALL} karakter")
    print(f"  Toplam karakter: {Fore.GREEN}{total_chars:,}{Style.RESET_ALL}")
    print()

    logger.info(
        f"Metin bolme tamamlandi: {len(documents)} dokuman -> "
        f"{len(chunks)} parca (ort. {avg_chunk_size:.0f} karakter)"
    )

    return chunks


def split_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """Ham metni parcalara bol.

    Args:
        text: Bolunecek metin.
        chunk_size: Parca boyutu. None ise config'ten alinir.
        chunk_overlap: Ortusme boyutu. None ise config'ten alinir.

    Returns:
        List[str]: Bolunmus metin parcalarinin listesi.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_text(text)
