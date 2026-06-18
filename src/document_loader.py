"""Doküman yükleme modülü.

PDF, TXT, DOCX ve CSV formatlarındaki dokümanları LangChain loader'ları ile yükler.
Belirtilen dizindeki tüm desteklenen dosyaları otomatik olarak tarar ve yükler.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from tqdm import tqdm
from colorama import Fore, Style

from src.config import config, logger


# Desteklenen dosya uzantıları ve karşılık gelen loader'lar
SUPPORTED_EXTENSIONS = {
 ".pdf": "PyPDFLoader",
 ".txt": "TextLoader",
 ".docx": "Docx2txtLoader",
 ".csv": "CSVLoader",
}


def _get_loader(file_path: str):
 """Dosya uzantısına göre uygun LangChain loader'ını döndür.

 Args:
 file_path: Yüklenecek dosyanın yolu.

 Returns:
 Uygun LangChain loader nesnesi.

 Raises:
 ValueError: Desteklenmeyen dosya formatı.
 """
 ext = Path(file_path).suffix.lower()

 if ext == ".pdf":
 from langchain_community.document_loaders import PyPDFLoader
 return PyPDFLoader(file_path)
 elif ext == ".txt":
 from langchain_community.document_loaders import TextLoader
 return TextLoader(file_path, encoding="utf-8")
 elif ext == ".docx":
 from langchain_community.document_loaders import Docx2txtLoader
 return Docx2txtLoader(file_path)
 elif ext == ".csv":
 from langchain_community.document_loaders import CSVLoader
 return CSVLoader(file_path, encoding="utf-8")
 else:
 raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")


def load_documents(
 directory: Optional[str] = None,
 file_types: Optional[List[str]] = None,
) -> List[Document]:
 """Belirtilen dizindeki tüm desteklenen dokümanları yükle.

 Args:
 directory: Dokümanların bulunduğu dizin. None ise config'ten alınır.
 file_types: Filtrelenecek dosya uzantıları (ör: ['.pdf', '.txt']).
 None ise tüm desteklenen formatlar yüklenir.

 Returns:
 List[Document]: Yüklenen dokümanların listesi.

 Example:
 >>> docs = load_documents("./documents")
 >>> len(docs)
 42
 """
 directory = directory or str(config.documents_dir)
 dir_path = Path(directory)

 if not dir_path.exists():
 logger.warning(f"Dizin bulunamadı: {directory}")
 print(f"{Fore.YELLOW} Dizin bulunamadı: {directory}{Style.RESET_ALL}")
 return []

 # Desteklenen dosyaları bul
 files = []
 for ext in SUPPORTED_EXTENSIONS:
 if file_types and ext not in file_types:
 continue
 files.extend(dir_path.glob(f"*{ext}"))

 if not files:
 logger.warning(f"Dizinde desteklenen dosya bulunamadı: {directory}")
 print(f"{Fore.YELLOW} Dizinde desteklenen dosya bulunamadı: {directory}{Style.RESET_ALL}")
 return []

 print(f"\n{Fore.CYAN} Doküman Yükleme Başlatılıyor...{Style.RESET_ALL}")
 print(f" Dizin: {directory}")
 print(f" Bulunan dosya sayısı: {Fore.GREEN}{len(files)}{Style.RESET_ALL}")
 print()

 all_documents: List[Document] = []
 loaded_count = 0
 error_count = 0
 total_chars = 0

 for file_path in tqdm(files, desc=" Dokümanlar yükleniyor", unit="dosya",
 bar_format="{l_bar}{bar:30}{r_bar}"):
 try:
 loader = _get_loader(str(file_path))
 docs = loader.load()

 # Her dokümana kaynak metadata ekle
 for doc in docs:
 doc.metadata["source_file"] = file_path.name
 doc.metadata["file_type"] = file_path.suffix.lower()
 total_chars += len(doc.page_content)

 all_documents.extend(docs)
 loaded_count += 1
 logger.info(f"Yüklendi: {file_path.name} ({len(docs)} sayfa/bölüm)")

 except Exception as e:
 error_count += 1
 logger.error(f"Dosya yüklenemedi: {file_path.name} - Hata: {str(e)}")
 print(f"\n{Fore.RED} Hata: {file_path.name} - {str(e)}{Style.RESET_ALL}")

 # Özet bilgileri göster
 print(f"\n{Fore.GREEN}{'─' * 40}")
 print(f" Yükleme Tamamlandı!")
 print(f"{'─' * 40}{Style.RESET_ALL}")
 print(f" Yüklenen dosya: {Fore.GREEN}{loaded_count}{Style.RESET_ALL}")
 print(f" Toplam doküman/sayfa: {Fore.GREEN}{len(all_documents)}{Style.RESET_ALL}")
 print(f" Toplam karakter: {Fore.GREEN}{total_chars:,}{Style.RESET_ALL}")
 if error_count > 0:
 print(f" Hatalı dosya: {Fore.RED}{error_count}{Style.RESET_ALL}")
 print()

 logger.info(
 f"Doküman yükleme tamamlandı: {loaded_count} dosya, "
 f"{len(all_documents)} doküman, {total_chars:,} karakter"
 )

 return all_documents


def load_single_document(file_path: str) -> List[Document]:
 """Tek bir dokümanı yükle.

 Args:
 file_path: Yüklenecek dosyanın yolu.

 Returns:
 List[Document]: Yüklenen dokümanların listesi.

 Raises:
 FileNotFoundError: Dosya bulunamazsa.
 ValueError: Desteklenmeyen dosya formatı.
 """
 path = Path(file_path)
 if not path.exists():
 raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

 if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
 raise ValueError(
 f"Desteklenmeyen dosya formatı: {path.suffix}. "
 f"Desteklenen: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
 )

 loader = _get_loader(str(path))
 docs = loader.load()

 for doc in docs:
 doc.metadata["source_file"] = path.name
 doc.metadata["file_type"] = path.suffix.lower()

 logger.info(f"Tek dosya yüklendi: {path.name} ({len(docs)} doküman)")
 return docs
