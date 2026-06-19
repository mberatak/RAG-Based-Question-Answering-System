"""Metin bolme modulu.

Yuklenen dokumanlari yapilandirilabilir boyutlarda parcalara (chunk) boler.
Desteklenen stratejiler:
  - recursive: RecursiveCharacterTextSplitter (varsayilan, tum formatlar)
  - header:    MarkdownHeaderTextSplitter (.md dosyalari icin baslik bazli bolme)
"""

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from colorama import Fore, Style

from src.config import config, logger


# Header-based splitting icin kullanilacak markdown baslik seviyeleri
_MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _split_recursive(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """RecursiveCharacterTextSplitter ile bolme (varsayilan strateji)."""
    separators = separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def _split_by_header(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Markdown basliklarına gore bolme (header stratejisi).

    .md dosyalari icin # ## ### basliklarini bolme noktasi olarak kullanir.
    Markdown olmayan dosyalar otomatik olarak recursive stratejiye duser.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )

    # Markdown olmayan dokumanlari ayir
    md_docs = [d for d in documents if d.metadata.get("file_type") == ".md"]
    other_docs = [d for d in documents if d.metadata.get("file_type") != ".md"]

    chunks: List[Document] = []

    # Markdown dokumanlari basliga gore bol
    for doc in md_docs:
        try:
            header_chunks = md_splitter.split_text(doc.page_content)
            # Metadata'yi kaynak dokumantan aktar
            for chunk in header_chunks:
                chunk.metadata.update({
                    k: v for k, v in doc.metadata.items()
                    if k not in chunk.metadata
                })
            chunks.extend(header_chunks)
        except Exception as e:
            logger.warning(f"Header split basarisiz, recursive'e donuluyor: {e}")
            chunks.extend(_split_recursive([doc], chunk_size, chunk_overlap))

    # Markdown olmayan dokumanlari recursive ile bol
    if other_docs:
        chunks.extend(_split_recursive(other_docs, chunk_size, chunk_overlap))

    # Buyuk header parcalarini chunk_size'a gore ikincil bölme
    if chunks:
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = secondary.split_documents(chunks)

    return chunks


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    strategy: Optional[str] = None,
    custom_separators: Optional[List[str]] = None,
) -> List[Document]:
    """Dokumanlari yapilandirilabilir boyutlarda parcalara bol.

    Args:
        documents: Bolunecek dokuman listesi.
        chunk_size: Parca boyutu (karakter). None ise config'ten alinir.
        chunk_overlap: Parcalar arasi ortusme. None ise config'ten alinir.
        strategy: Bolme stratejisi ('recursive' | 'header').
                  None ise config.chunk_strategy kullanilir.
        custom_separators: Sadece recursive stratejide gecerli ozel ayrac listesi.

    Returns:
        List[Document]: Bolunmus dokuman parcalarinin listesi.

    Notes:
        Geriye donuk uyumluluk korunur — strateji belirtilmezse mevcut
        davranis (recursive) degismez.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    strategy = (strategy or config.chunk_strategy).lower()

    print(f"\n{Fore.CYAN} Metin Bolme Baslatiliyor...{Style.RESET_ALL}")
    print(f"  Strateji  : {strategy}")
    print(f"  Chunk boyutu: {chunk_size}")
    print(f"  Chunk ortusme: {chunk_overlap}")
    print(f"  Dokuman sayisi: {len(documents)}")

    if strategy == "header":
        chunks = _split_by_header(documents, chunk_size, chunk_overlap)
    else:
        # varsayilan: recursive
        chunks = _split_recursive(documents, chunk_size, chunk_overlap, custom_separators)

    # Her chunk'a ek metadata ekle
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["chunk_strategy"] = strategy

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
        f"Metin bolme tamamlandi ({strategy}): {len(documents)} dokuman -> "
        f"{len(chunks)} parca (ort. {avg_chunk_size:.0f} karakter)"
    )

    return chunks


def split_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """Ham metni parcalara bol (her zaman recursive strateji kullanir).

    Args:
        text: Bolunecek metin.
        chunk_size: Parca boyutu. None ise config'ten alinir.
        chunk_overlap: Ortusme boyutu. None ise config'ten alinir.

    Returns:
        List[str]: Bolunmus metin parcalarinin listesi.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return splitter.split_text(text)
