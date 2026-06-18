"""Konfigürasyon modülü.

Tüm uygulama ayarlarını merkezi olarak yönetir.
.env dosyasından API anahtarlarını okur ve logging yapılandırmasını sağlar.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from colorama import init as colorama_init, Fore, Style

# Colorama'yı başlat (Windows uyumluluğu için)
colorama_init(autoreset=True)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# .env dosyasını yükle
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
 """Uygulama konfigürasyon sınıfı.

 Attributes:
 gemini_api_key: Google Gemini API anahtarı.
 embedding_model: Embedding modeli adı.
 llm_model: Cevap üretme modeli adı.
 temperature: LLM sıcaklık değeri (0.0 - 1.0).
 chunk_size: Metin parça boyutu (karakter).
 chunk_overlap: Parçalar arası örtüşme (karakter).
 top_k: Döndürülecek en benzer doküman sayısı.
 vector_db: Vektör veritabanı seçimi ('faiss' veya 'chroma').
 log_level: Loglama seviyesi.
 documents_dir: Doküman dizini yolu.
 index_dir: Vektör indeks dizini yolu.
 cache_dir: Önbellek dizini yolu.
 logs_dir: Log dizini yolu.
 cache_ttl_hours: Önbellek geçerlilik süresi (saat).
 """

 # API Ayarları
 gemini_api_key: str = ""

 # Model Ayarları
 embedding_model: str = "gemini-embedding-001"
 llm_model: str = "gemini-2.5-flash"
 temperature: float = 0.3

 # Metin Bölme Ayarları
 chunk_size: int = 1000
 chunk_overlap: int = 200

 # Arama Ayarları
 top_k: int = 5

 # Vektör Veritabanı
 vector_db: str = "faiss" # 'faiss' veya 'chroma'

 # Loglama
 log_level: str = "INFO"

 # Dizinler
 documents_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "documents")
 index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "index")
 cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "cache")
 logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

 # Önbellek
 cache_ttl_hours: int = 24

 # Embedding Maliyet (USD / 1K token)
 embedding_cost_per_1k_tokens: float = 0.0001

 def __post_init__(self):
 """Ortam değişkenlerinden ayarları yükle ve dizinleri oluştur."""
 # Ortam değişkenlerinden oku (varsa)
 self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
 self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
 self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
 self.temperature = float(os.getenv("TEMPERATURE", str(self.temperature)))
 self.chunk_size = int(os.getenv("CHUNK_SIZE", str(self.chunk_size)))
 self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(self.chunk_overlap)))
 self.top_k = int(os.getenv("TOP_K", str(self.top_k)))
 self.vector_db = os.getenv("VECTOR_DB", self.vector_db).lower()
 self.log_level = os.getenv("LOG_LEVEL", self.log_level).upper()

 # Dizinleri oluştur
 for dir_path in [self.documents_dir, self.index_dir, self.cache_dir, self.logs_dir]:
 dir_path.mkdir(parents=True, exist_ok=True)

 def validate(self) -> bool:
 """Konfigürasyon ayarlarını doğrula.

 Returns:
 bool: Ayarlar geçerliyse True.

 Raises:
 ValueError: API anahtarı eksikse veya ayarlar geçersizse.
 """
 if not self.gemini_api_key:
 raise ValueError(
 f"{Fore.RED}HATA: GEMINI_API_KEY ayarlanmamış!{Style.RESET_ALL}\n"
 f"Lütfen .env dosyasına geçerli bir Gemini API anahtarı ekleyin.\n"
 f"Örnek: GEMINI_API_KEY=AIzaSy..."
 )
 if self.vector_db not in ("faiss", "chroma"):
 raise ValueError(
 f"{Fore.RED}HATA: VECTOR_DB '{self.vector_db}' geçersiz. "
 f"'faiss' veya 'chroma' olmalı.{Style.RESET_ALL}"
 )
 if self.chunk_size <= 0:
 raise ValueError("chunk_size pozitif bir sayı olmalıdır.")
 if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
 raise ValueError("chunk_overlap 0 ile chunk_size arasında olmalıdır.")
 return True

 def display(self) -> None:
 """Aktif konfigürasyon ayarlarını renkli formatta göster."""
 print(f"\n{Fore.CYAN}{'=' * 50}")
 print(f" Aktif Konfigürasyon")
 print(f"{'=' * 50}{Style.RESET_ALL}")
 print(f" {Fore.YELLOW}LLM Model:{Style.RESET_ALL} {self.llm_model}")
 print(f" {Fore.YELLOW}Embedding Model:{Style.RESET_ALL} {self.embedding_model}")
 print(f" {Fore.YELLOW}Temperature:{Style.RESET_ALL} {self.temperature}")
 print(f" {Fore.YELLOW}Chunk Size:{Style.RESET_ALL} {self.chunk_size}")
 print(f" {Fore.YELLOW}Chunk Overlap:{Style.RESET_ALL} {self.chunk_overlap}")
 print(f" {Fore.YELLOW}Top-K:{Style.RESET_ALL} {self.top_k}")
 print(f" {Fore.YELLOW}Vektör DB:{Style.RESET_ALL} {self.vector_db}")
 print(f" {Fore.YELLOW}Log Seviyesi:{Style.RESET_ALL} {self.log_level}")
 print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")


def setup_logging(config: Config) -> logging.Logger:
 """Loglama yapılandırmasını ayarla.

 Args:
 config: Uygulama konfigürasyonu.

 Returns:
 logging.Logger: Yapılandırılmış logger nesnesi.
 """
 logger = logging.getLogger("rag_system")
 logger.setLevel(getattr(logging, config.log_level, logging.INFO))

 # Dosya handler'ı
 log_file = config.logs_dir / "app.log"
 file_handler = logging.FileHandler(log_file, encoding="utf-8")
 file_handler.setLevel(logging.DEBUG)
 file_formatter = logging.Formatter(
 "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
 datefmt="%Y-%m-%d %H:%M:%S",
 )
 file_handler.setFormatter(file_formatter)

 # Konsol handler'ı
 console_handler = logging.StreamHandler()
 console_handler.setLevel(getattr(logging, config.log_level, logging.INFO))
 console_formatter = logging.Formatter(
 f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - %(levelname)s - %(message)s",
 datefmt="%H:%M:%S",
 )
 console_handler.setFormatter(console_formatter)

 # Mevcut handler'ları temizle ve yenilerini ekle
 logger.handlers.clear()
 logger.addHandler(file_handler)
 logger.addHandler(console_handler)

 return logger


# Global konfigürasyon ve logger
config = Config()
logger = setup_logging(config)
