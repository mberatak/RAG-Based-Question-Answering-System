"""Metin bölme modülü.

Yüklenen dokümanları yapılandırılabilir boyutlarda parçalara (chunk) böler.
RecursiveCharacterTextSplitter kullanarak anlamlı bölme noktaları oluşturur.
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
 """Dokümanları yapılandırılabilir boyutlarda parçalara böl.

 Args:
 documents: Bölünecek doküman listesi.
 chunk_size: Parça boyutu (karakter). None ise config'ten alınır.
 chunk_overlap: Parçalar arası örtüşme. None ise config'ten alınır.
 custom_separators: Özel ayraç listesi. None ise varsayılanlar kullanılır.
 Örnek: ["###", "\\n\\n", "\\n", " "]

 Returns:
 List[Document]: Bölünmüş doküman parçalarının listesi.

 Example:
 >>> chunks = split_documents(documents, chunk_size=500)
 >>> len(chunks)
 128
 """
 chunk_size = chunk_size if chunk_size is not None else config.chunk_size
 chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

 # Varsayılan veya özel ayraçlar
 separators = custom_separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

 print(f"\n{Fore.CYAN}️ Metin Bölme Başlatılıyor...{Style.RESET_ALL}")
 print(f" Chunk boyutu: {chunk_size}")
 print(f" Chunk örtüşme: {chunk_overlap}")
 print(f" Doküman sayısı: {len(documents)}")

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

 # Kaynak dosya adını kısa formata dönüştür
 if "source" in chunk.metadata:
 source = chunk.metadata["source"]
 # Sadece dosya adını al (tam yolu değil)
 chunk.metadata["source_short"] = source.split("\\")[-1].split("/")[-1]

 # İstatistikleri hesapla
 total_chars = sum(len(c.page_content) for c in chunks)
 avg_chunk_size = total_chars / len(chunks) if chunks else 0

 print(f"\n{Fore.GREEN}{'─' * 40}")
 print(f" Metin Bölme Tamamlandı!")
 print(f"{'─' * 40}{Style.RESET_ALL}")
 print(f" Toplam parça sayısı: {Fore.GREEN}{len(chunks)}{Style.RESET_ALL}")
 print(f" Ortalama parça boyutu: {Fore.GREEN}{avg_chunk_size:.0f}{Style.RESET_ALL} karakter")
 print(f" Toplam karakter: {Fore.GREEN}{total_chars:,}{Style.RESET_ALL}")
 print()

 logger.info(
 f"Metin bölme tamamlandı: {len(documents)} doküman → "
 f"{len(chunks)} parça (ort. {avg_chunk_size:.0f} karakter)"
 )

 return chunks


def split_text(
 text: str,
 chunk_size: Optional[int] = None,
 chunk_overlap: Optional[int] = None,
) -> List[str]:
 """Ham metni parçalara böl.

 Args:
 text: Bölünecek metin.
 chunk_size: Parça boyutu. None ise config'ten alınır.
 chunk_overlap: Örtüşme boyutu. None ise config'ten alınır.

 Returns:
 List[str]: Bölünmüş metin parçalarının listesi.
 """
 chunk_size = chunk_size if chunk_size is not None else config.chunk_size
 chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

 text_splitter = RecursiveCharacterTextSplitter(
 chunk_size=chunk_size,
 chunk_overlap=chunk_overlap,
 length_function=len,
 )

 return text_splitter.split_text(text)
