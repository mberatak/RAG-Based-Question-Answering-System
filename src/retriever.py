"""Arama ve getirme modulu.

Semantic search ve hybrid search (BM25 + semantic) ile en ilgili dokuman
parcalarini bulur. Benzerlik skorlari ve metadata filtreleme destegi icerir.
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from colorama import Fore, Style

from src.config import config, logger


def semantic_search(
    vector_store,
    query: str,
    top_k: Optional[int] = None,
    file_filter: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Anlamsal arama ile en benzer dokuman parcalarini bul.

    Args:
        vector_store: Vektor deposu nesnesi.
        query: Arama sorgusu.
        top_k: Dondurulecek sonuc sayisi. None ise config'ten alinir.
        file_filter: Dosya turu filtresi (or: '.pdf', '.txt').
                     None ise tum dosyalarda aranir.

    Returns:
        List[Tuple[Document, float]]: (dokuman, benzerlik_skoru) ciftlerinin listesi.
    """
    top_k = top_k or config.top_k

    fetch_k = top_k * 3 if file_filter else top_k

    results = vector_store.similarity_search_with_score(
        query=query,
        k=fetch_k,
    )

    if file_filter:
        results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("file_type", "") == file_filter
        ][:top_k]

    logger.info(
        f"Semantic search: '{query[:50]}...' -> {len(results)} sonuc "
        f"(top_k={top_k}, filtre={file_filter})"
    )

    return results


def hybrid_search(
    vector_store,
    query: str,
    chunks: List[Document],
    top_k: Optional[int] = None,
    semantic_weight: float = 0.5,
) -> List[Tuple[Document, float]]:
    """Hibrit arama (BM25 + Semantic) ile en benzer dokuman parcalarini bul.

    Args:
        vector_store: Vektor deposu nesnesi.
        query: Arama sorgusu.
        chunks: Tum dokuman parcalari (BM25 icin gerekli).
        top_k: Dondurulecek sonuc sayisi. None ise config'ten alinir.
        semantic_weight: Semantic arama agirligi (0.0-1.0).

    Returns:
        List[Tuple[Document, float]]: (dokuman, birlesik_skor) ciftlerinin listesi.
    """
    from rank_bm25 import BM25Okapi

    top_k = top_k or config.top_k

    corpus = [doc.page_content for doc in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_normalized = {i: score / max_bm25 for i, score in enumerate(bm25_scores)}

    semantic_results = vector_store.similarity_search_with_score(
        query=query,
        k=min(len(chunks), top_k * 3),
    )

    semantic_map = {}
    if semantic_results:
        max_dist = max(score for _, score in semantic_results) or 1
        for doc, score in semantic_results:
            similarity = 1 - (score / max_dist) if max_dist > 0 else 0
            content_key = doc.page_content[:100]
            semantic_map[content_key] = (doc, similarity)

    combined_scores = []
    for i, chunk in enumerate(chunks):
        content_key = chunk.page_content[:100]
        bm25_score = bm25_normalized.get(i, 0)
        semantic_score = 0

        if content_key in semantic_map:
            _, semantic_score = semantic_map[content_key]

        hybrid_score = (semantic_weight * semantic_score +
                        (1 - semantic_weight) * bm25_score)
        combined_scores.append((chunk, hybrid_score))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    results = combined_scores[:top_k]

    logger.info(
        f"Hybrid search: '{query[:50]}...' -> {len(results)} sonuc "
        f"(semantic_weight={semantic_weight})"
    )

    return results


def display_search_results(
    results: List[Tuple[Document, float]],
    search_type: str = "semantic",
) -> None:
    """Arama sonuclarini renkli formatta goster.

    Args:
        results: (dokuman, skor) ciftlerinin listesi.
        search_type: Arama tipi ('semantic' veya 'hybrid').
    """
    print(f"\n{Fore.CYAN} Arama Sonuclari ({search_type.capitalize()}){Style.RESET_ALL}")
    print(f"{'-' * 60}")

    if not results:
        print(f"  {Fore.YELLOW}Sonuc bulunamadi.{Style.RESET_ALL}")
        return

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source_short", "Bilinmiyor"))

        if search_type == "semantic":
            score_color = Fore.GREEN if score < 0.5 else (Fore.YELLOW if score < 1.0 else Fore.RED)
            score_label = f"Mesafe: {score:.4f}"
        else:
            score_color = Fore.GREEN if score > 0.7 else (Fore.YELLOW if score > 0.3 else Fore.RED)
            score_label = f"Skor: {score:.4f}"

        print(f"\n  {Fore.WHITE}[{i}]{Style.RESET_ALL} {score_color}{score_label}{Style.RESET_ALL}")
        print(f"  Kaynak: {Fore.BLUE}{source}{Style.RESET_ALL}")

        preview = doc.page_content[:200].replace("\n", " ").strip()
        print(f"  {preview}...")

    print(f"\n{'-' * 60}")


def get_retriever(
    vector_store,
    search_type: str = "similarity",
    top_k: Optional[int] = None,
):
    """Vektor deposundan LangChain retriever nesnesi olustur.

    Args:
        vector_store: Vektor deposu nesnesi.
        search_type: Arama tipi ('similarity' veya 'mmr').
        top_k: Dondurulecek sonuc sayisi.

    Returns:
        LangChain Retriever nesnesi.
    """
    top_k = top_k or config.top_k

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k},
    )

    logger.info(f"Retriever olusturuldu: type={search_type}, k={top_k}")

    return retriever
