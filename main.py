"""RAG Tabanlı Soru-Cevap Sistemi — Ana Giriş Noktası.

Komut satırından etkileşimli olarak çalışan ana uygulama.
Doküman yükleme, indeksleme, sorgulama ve değerlendirme işlemlerini yönetir.

Kullanım:
    python main.py --ingest              # Dokümanları yükle ve indeksle
    python main.py --query "sorunuz"     # Tek soru sor
    python main.py --interactive         # Etkileşimli mod
    python main.py --rebuild             # İndeksi yeniden oluştur
    python main.py --evaluate            # Test soruları ile değerlendir
"""

import argparse
import sys
import time
from pathlib import Path

from colorama import Fore, Style, init as colorama_init

# Colorama başlat
colorama_init(autoreset=True)

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config, logger
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import create_vector_store, save_vector_store, load_vector_store
from src.retriever import semantic_search, hybrid_search, display_search_results
from src.generator import create_qa_chain, ask_question
from src.evaluator import run_full_evaluation, load_test_questions
from src.cache import QueryCache


# Banner
BANNER = f"""
{Fore.CYAN}+----------------------------------------------------------+
|                                                          |
|   RAG Tabanlı Soru-Cevap Sistemi  v1.0                  |
|   LangChain + Gemini + FAISS/ChromaDB                    |
|                                                          |
+----------------------------------------------------------+{Style.RESET_ALL}
"""


def ingest_documents(source_dir: str = None) -> object:
    """Dokümanları yükle, böl ve vektör deposuna indeksle.

    Args:
        source_dir: Doküman dizini. None ise config'ten alınır.

    Returns:
        Oluşturulan vektör deposu nesnesi.
    """
    start_time = time.time()

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"  DOKUMAN YUKLEME VE INDEKSLEME")
    print(f"{'=' * 60}{Style.RESET_ALL}")

    # 1. Dokumanlari yukle
    documents = load_documents(source_dir)
    if not documents:
        print(f"{Fore.RED}Hata: Yuklenecek dokuman bulunamadi!{Style.RESET_ALL}")
        print(f"   Dokumanlarinizi su dizine koyun: {config.documents_dir}")
        sys.exit(1)

    # 2. Metni bol
    chunks = split_documents(documents)

    # 3. Vektor deposu olustur
    vector_store = create_vector_store(chunks)

    # 4. Diske kaydet
    save_vector_store(vector_store)

    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}{'=' * 60}")
    print(f"  INDEKSLEME TAMAMLANDI")
    print(f"  Sure: {elapsed:.1f}s")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    logger.info(f"İndeksleme tamamlandı: {elapsed:.1f}s")

    return vector_store


def query_single(question: str, search_type: str = "semantic") -> None:
    """Tek bir soru sor ve cevap al.

    Args:
        question: Sorulacak soru.
        search_type: Arama tipi ('semantic' veya 'hybrid').
    """
    # Önbellek kontrolü
    cache = QueryCache()
    cached = cache.get(question)
    if cached:
        print(f"\n{Fore.GREEN}Cevap (onbellekten):{Style.RESET_ALL}")
        print(f"   {cached['answer']}")
        if cached.get("sources"):
            print(f"\n{Fore.CYAN}Kaynaklar:{Style.RESET_ALL}")
            for i, src in enumerate(cached["sources"], 1):
                print(f"   [{i}] {src['source']}")
        return

    # Vektor deposunu yukle
    vector_store = load_vector_store()
    if not vector_store:
        print(f"{Fore.RED}Hata: Vektor deposu bulunamadi. "
              f"Once 'python main.py --ingest' calistirin.{Style.RESET_ALL}")
        sys.exit(1)

    # Arama sonuçlarını göster
    if search_type == "hybrid":
        # Hybrid search için tüm chunk'ları yüklemek gerekir (basit implementasyon)
        documents = load_documents()
        chunks = split_documents(documents)
        results = hybrid_search(vector_store, question, chunks)
        display_search_results(results, search_type="hybrid")
    else:
        results = semantic_search(vector_store, question)
        display_search_results(results, search_type="semantic")

    # Cevap üret
    qa_chain = create_qa_chain(vector_store)
    result = ask_question(qa_chain, question)

    # Önbelleğe kaydet
    cache.set(question, result)


def interactive_mode(search_type: str = "semantic") -> None:
    """Etkileşimli soru-cevap modu.

    Args:
        search_type: Varsayılan arama tipi.
    """
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"  ETKILESIMLI MOD")
    print(f"  Cikis: 'q' veya 'cik'")
    print(f"  Onbellegi temizle: 'temizle'")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    # Vektor deposunu yukle
    vector_store = load_vector_store()
    if not vector_store:
        print(f"{Fore.RED}Hata: Vektor deposu bulunamadi. "
              f"Once 'python main.py --ingest' calistirin.{Style.RESET_ALL}")
        sys.exit(1)

    qa_chain = create_qa_chain(vector_store)
    cache = QueryCache()
    question_count = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        try:
            question = input(f"\n{Fore.WHITE}Soru: {Style.RESET_ALL}").strip()

            if not question:
                continue

            if question.lower() in ("q", "quit", "exit", "cik", "kapat"):
                print(f"\n{Fore.CYAN}Oturum Ozeti:{Style.RESET_ALL}")
                print(f"   Soru sayisi:  {question_count}")
                print(f"   Toplam token: {total_tokens:,}")
                break

            if question.lower() in ("temizle", "clear"):
                cache.clear()
                continue

            if question.lower() in ("istatistik", "stats"):
                cache.stats()
                continue

            # Onbellek kontrolu
            cached = cache.get(question)
            if cached:
                print(f"\n{Fore.GREEN}Cevap (onbellekten):{Style.RESET_ALL}")
                print(f"   {cached['answer']}")
                question_count += 1
                continue

            # Cevap uret
            result = ask_question(qa_chain, question)
            question_count += 1
            total_tokens += result["total_tokens"]
            total_cost += result["total_cost"]

            # Onbellege kaydet
            cache.set(question, result)

        except KeyboardInterrupt:
            print(f"\n")
            break
        except Exception as e:
            logger.error(f"Hata: {str(e)}")
            print(f"{Fore.RED}Hata: {str(e)}{Style.RESET_ALL}")


def rebuild_index() -> None:
    """Vektor indeksini yeniden olustur."""
    print(f"\n{Fore.YELLOW}Indeks yeniden olusturuluyor...{Style.RESET_ALL}")
    print(f"   Mevcut indeks silinecek ve yeniden olusturulacak.\n")

    import shutil
    index_dir = config.index_dir
    if index_dir.exists():
        for item in index_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"   {Fore.YELLOW}Eski indeks silindi.{Style.RESET_ALL}")

    ingest_documents()


def main():
    """Ana uygulama giriş noktası."""
    parser = argparse.ArgumentParser(
        description="RAG Tabanlı Soru-Cevap Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım örnekleri:
  python main.py --ingest              Dokümanları yükle ve indeksle
  python main.py --query "sorunuz"     Tek soru sor
  python main.py --interactive         Etkileşimli mod
  python main.py --rebuild             İndeksi yeniden oluştur
  python main.py --evaluate            Test soruları ile değerlendir
  python main.py --ingest --source sample_docs   Belirli dizinden yükle
        """,
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Dokümanları yükle ve vektör deposuna indeksle",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Tek bir soru sor",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Etkileşimli soru-cevap modu",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Vektör indeksini yeniden oluştur",
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Test soruları ile değerlendirme çalıştır",
    )
    parser.add_argument(
        "--search-type",
        choices=["semantic", "hybrid"],
        default="semantic",
        help="Arama tipi (varsayılan: semantic)",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Doküman kaynak dizini (varsayılan: documents/)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Aktif konfigürasyonu göster",
    )

    args = parser.parse_args()

    # Banner göster
    print(BANNER)

    # Hiçbir argüman verilmemişse yardım göster
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Konfigürasyon göster
    if args.config:
        config.display()
        return

    # API anahtarı kontrolü (evaluate ve config dışında gerekli)
    if not args.config:
        try:
            config.validate()
        except ValueError as e:
            print(str(e))
            sys.exit(1)

    config.display()

    # İşlemleri çalıştır
    if args.rebuild:
        rebuild_index()

    elif args.ingest:
        source = args.source or str(config.documents_dir)
        ingest_documents(source)

    elif args.query:
        query_single(args.query, search_type=args.search_type)

    elif args.interactive:
        interactive_mode(search_type=args.search_type)

    elif args.evaluate:
        vector_store = load_vector_store()
        if not vector_store:
            print(f"{Fore.RED}❌ Vektör deposu bulunamadı! "
                  f"Önce 'python main.py --ingest' çalıştırın.{Style.RESET_ALL}")
            sys.exit(1)
        run_full_evaluation(vector_store)


if __name__ == "__main__":
    main()
