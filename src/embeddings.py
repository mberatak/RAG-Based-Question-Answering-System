"""Vektör embedding ve depolama modülü.

Google Gemini embedding modeli ile vektör oluşturur ve FAISS veya ChromaDB ile indeksler.
Embedding maliyet hesaplama, diske kaydetme ve yükleme fonksiyonlarını içerir.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm
from colorama import Fore, Style

from src.config import config, logger


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
 """Google Gemini embedding modelini oluştur ve döndür.

 Returns:
 GoogleGenerativeAIEmbeddings: Yapılandırılmış embedding modeli.
 """
 return GoogleGenerativeAIEmbeddings(
 model=config.embedding_model,
 google_api_key=config.gemini_api_key,
 task_type="retrieval_document",
 )


def estimate_embedding_cost(chunks: List[Document]) -> Tuple[int, float]:
 """Embedding oluşturma için tahmini karakter/token sayısını göster.

 Args:
 chunks: Embedding oluşturulacak doküman parçaları.

 Returns:
 Tuple[int, float]: (toplam_karakter_sayısı, maliyet) — Gemini ücretsiz kota.

 Example:
 >>> chars, cost = estimate_embedding_cost(chunks)
 >>> print(f"Toplam: {chars} karakter")
 """
 total_chars = sum(len(chunk.page_content) for chunk in chunks)
 # Gemini ücretsiz kotada maliyet yok
 cost = 0.0

 print(f"\n{Fore.CYAN} Embedding Bilgisi{Style.RESET_ALL}")
 print(f" Toplam karakter: {Fore.YELLOW}{total_chars:,}{Style.RESET_ALL}")
 print(f" Model: {config.embedding_model}")
 print(f" Maliyet: {Fore.GREEN}Ücretsiz (Gemini kota){Style.RESET_ALL}")
 print()

 logger.info(f"Embedding tahmini: {total_chars:,} karakter, Gemini modeli")

 return total_chars, cost


def create_vector_store(
 chunks: List[Document],
 vector_db: Optional[str] = None,
) -> object:
 """Doküman parçalarından vektör deposu oluştur.

 Args:
 chunks: Embedding oluşturulacak doküman parçaları.
 vector_db: Kullanılacak vektör veritabanı ('faiss' veya 'chroma').
 None ise config'ten alınır.

 Returns:
 Oluşturulan vektör deposu nesnesi (FAISS veya Chroma).

 Raises:
 ValueError: Geçersiz vector_db değeri.
 """
 vector_db = vector_db or config.vector_db
 embeddings = get_embeddings()

 # Maliyet tahminini göster
 estimate_embedding_cost(chunks)

 print(f"{Fore.CYAN} Vektör deposu oluşturuluyor ({vector_db.upper()})...{Style.RESET_ALL}")
 logger.info(f"Vektör deposu oluşturuluyor: {len(chunks)} parça, DB: {vector_db}")

 if vector_db == "faiss":
 from langchain_community.vectorstores import FAISS

 # FAISS ile batch embedding
 vector_store = FAISS.from_documents(
 documents=chunks,
 embedding=embeddings,
 )

 elif vector_db == "chroma":
 from langchain_community.vectorstores import Chroma

 persist_dir = str(config.index_dir / "chroma_db")
 vector_store = Chroma.from_documents(
 documents=chunks,
 embedding=embeddings,
 persist_directory=persist_dir,
 )

 else:
 raise ValueError(
 f"Geçersiz vektör veritabanı: '{vector_db}'. "
 f"'faiss' veya 'chroma' kullanın."
 )

 print(f"{Fore.GREEN} Vektör deposu başarıyla oluşturuldu!{Style.RESET_ALL}")
 print(f" Parça sayısı: {len(chunks)}")
 print()

 logger.info(f"Vektör deposu oluşturuldu: {len(chunks)} parça ({vector_db})")

 return vector_store


def save_vector_store(vector_store, vector_db: Optional[str] = None) -> str:
 """Vektör deposunu diske kaydet.

 Args:
 vector_store: Kaydedilecek vektör deposu nesnesi.
 vector_db: Vektör veritabanı tipi. None ise config'ten alınır.

 Returns:
 str: Kaydedilen dizinin yolu.
 """
 vector_db = vector_db or config.vector_db
 save_path = str(config.index_dir / f"{vector_db}_index")

 if vector_db == "faiss":
 vector_store.save_local(save_path)
 elif vector_db == "chroma":
 # Chroma otomatik olarak persist eder
 pass

 print(f"{Fore.GREEN} Vektör deposu kaydedildi: {save_path}{Style.RESET_ALL}")
 logger.info(f"Vektör deposu kaydedildi: {save_path}")

 return save_path


def load_vector_store(vector_db: Optional[str] = None) -> Optional[object]:
 """Vektör deposunu diskten yükle.

 Args:
 vector_db: Vektör veritabanı tipi. None ise config'ten alınır.

 Returns:
 Yüklenen vektör deposu nesnesi veya None (bulunamazsa).
 """
 vector_db = vector_db or config.vector_db
 embeddings = get_embeddings()

 if vector_db == "faiss":
 load_path = str(config.index_dir / "faiss_index")
 index_file = Path(load_path) / "index.faiss"

 if not index_file.exists():
 logger.warning(f"FAISS indeksi bulunamadı: {load_path}")
 print(f"{Fore.YELLOW} FAISS indeksi bulunamadı. "
 f"Önce '--ingest' ile indeks oluşturun.{Style.RESET_ALL}")
 return None

 from langchain_community.vectorstores import FAISS

 vector_store = FAISS.load_local(
 load_path,
 embeddings,
 allow_dangerous_deserialization=True,
 )

 elif vector_db == "chroma":
 persist_dir = str(config.index_dir / "chroma_db")

 if not Path(persist_dir).exists():
 logger.warning(f"ChromaDB dizini bulunamadı: {persist_dir}")
 print(f"{Fore.YELLOW} ChromaDB indeksi bulunamadı. "
 f"Önce '--ingest' ile indeks oluşturun.{Style.RESET_ALL}")
 return None

 from langchain_community.vectorstores import Chroma

 vector_store = Chroma(
 persist_directory=persist_dir,
 embedding_function=embeddings,
 )

 else:
 raise ValueError(f"Geçersiz vektör veritabanı: '{vector_db}'")

 print(f"{Fore.GREEN} Vektör deposu yüklendi ({vector_db.upper()}){Style.RESET_ALL}")
 logger.info(f"Vektör deposu yüklendi: {vector_db}")

 return vector_store
