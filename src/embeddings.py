"""Vektor embedding ve depolama modulu.

Google Gemini embedding modeli ile vektor olusturur ve FAISS veya ChromaDB ile indeksler.
Embedding maliyet hesaplama, diske kaydetme ve yukleme fonksiyonlarini icerir.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from colorama import Fore, Style

from src.config import config, logger


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Google Gemini embedding modelini olustur ve dondur.

    Returns:
        GoogleGenerativeAIEmbeddings: Yapilandirilmis embedding modeli.
    """
    return GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=config.gemini_api_key,
        task_type="retrieval_document",
    )


def estimate_embedding_cost(chunks: List[Document]) -> Tuple[int, float]:
    """Embedding olusturma icin tahmini karakter/token sayisini goster.

    Args:
        chunks: Embedding olusturulacak dokuman parcalari.

    Returns:
        Tuple[int, float]: (toplam_karakter_sayisi, maliyet)
    """
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    cost = 0.0

    print(f"\n{Fore.CYAN} Embedding Bilgisi{Style.RESET_ALL}")
    print(f"  Toplam karakter: {Fore.YELLOW}{total_chars:,}{Style.RESET_ALL}")
    print(f"  Model: {config.embedding_model}")
    print(f"  Maliyet: {Fore.GREEN}Ucretsiz (Gemini kota){Style.RESET_ALL}")
    print()

    logger.info(f"Embedding tahmini: {total_chars:,} karakter, Gemini modeli")

    return total_chars, cost


def create_vector_store(
    chunks: List[Document],
    vector_db: Optional[str] = None,
) -> object:
    """Dokuman parcalarindan vektor deposu olustur.

    Args:
        chunks: Embedding olusturulacak dokuman parcalari.
        vector_db: Kullanilacak vektor veritabani ('faiss' veya 'chroma').
                   None ise config'ten alinir.

    Returns:
        Olusturulan vektor deposu nesnesi (FAISS veya Chroma).

    Raises:
        ValueError: Gecersiz vector_db degeri.
    """
    vector_db = vector_db or config.vector_db
    embeddings = get_embeddings()

    estimate_embedding_cost(chunks)

    print(f"{Fore.CYAN} Vektor deposu olusturuluyor ({vector_db.upper()})...{Style.RESET_ALL}")
    logger.info(f"Vektor deposu olusturuluyor: {len(chunks)} parca, DB: {vector_db}")

    if vector_db == "faiss":
        from langchain_community.vectorstores import FAISS

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
            f"Gecersiz vektor veritabani: '{vector_db}'. "
            f"'faiss' veya 'chroma' kullanin."
        )

    print(f"{Fore.GREEN} Vektor deposu basariyla olusturuldu!{Style.RESET_ALL}")
    print(f"  Parca sayisi: {len(chunks)}")
    print()

    logger.info(f"Vektor deposu olusturuldu: {len(chunks)} parca ({vector_db})")

    return vector_store


def save_vector_store(vector_store, vector_db: Optional[str] = None) -> str:
    """Vektor deposunu diske kaydet.

    Args:
        vector_store: Kaydedilecek vektor deposu nesnesi.
        vector_db: Vektor veritabani tipi. None ise config'ten alinir.

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

    print(f"{Fore.GREEN} Vektor deposu kaydedildi: {save_path}{Style.RESET_ALL}")
    logger.info(f"Vektor deposu kaydedildi: {save_path}")

    return save_path


def load_vector_store(vector_db: Optional[str] = None) -> Optional[object]:
    """Vektor deposunu diskten yukle.

    Args:
        vector_db: Vektor veritabani tipi. None ise config'ten alinir.

    Returns:
        Yuklenen vektor deposu nesnesi veya None (bulunamazsa).
    """
    vector_db = vector_db or config.vector_db
    embeddings = get_embeddings()

    if vector_db == "faiss":
        load_path = str(config.index_dir / "faiss_index")
        index_file = Path(load_path) / "index.faiss"

        if not index_file.exists():
            logger.warning(f"FAISS indeksi bulunamadi: {load_path}")
            print(f"{Fore.YELLOW} FAISS indeksi bulunamadi. "
                  f"Once '--ingest' ile indeks olusturun.{Style.RESET_ALL}")
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
            logger.warning(f"ChromaDB dizini bulunamadi: {persist_dir}")
            print(f"{Fore.YELLOW} ChromaDB indeksi bulunamadi. "
                  f"Once '--ingest' ile indeks olusturun.{Style.RESET_ALL}")
            return None

        from langchain_community.vectorstores import Chroma

        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    else:
        raise ValueError(f"Gecersiz vektor veritabani: '{vector_db}'")

    print(f"{Fore.GREEN} Vektor deposu yuklendi ({vector_db.upper()}){Style.RESET_ALL}")
    logger.info(f"Vektor deposu yuklendi: {vector_db}")

    return vector_store
