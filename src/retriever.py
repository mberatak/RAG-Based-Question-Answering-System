"""Arama ve getirme modülü.

Semantic search ve hybrid search (BM25 + semantic) ile en ilgili doküman
parçalarını bulur. Benzerlik skorları ve metadata filtreleme desteği içerir.
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
 """Anlamsal arama ile en benzer doküman parçalarını bul.

 Args:
 vector_store: Vektör deposu nesnesi.
 query: Arama sorgusu.
 top_k: Döndürülecek sonuç sayısı. None ise config'ten alınır.
 file_filter: Dosya türü filtresi (ör: '.pdf', '.txt').
 None ise tüm dosyalarda aranır.

 Returns:
 List[Tuple[Document, float]]: (doküman, benzerlik_skoru) çiftlerinin listesi.
 Skor 0'a yakınsa çok benzer, büyükse az benzer demektir.

 Example:
 >>> results = semantic_search(store, "Ankara'nın nüfusu nedir?")
 >>> for doc, score in results:
 ... print(f"Skor: {score:.4f} | {doc.page_content[:50]}")
 """
 top_k = top_k or config.top_k

 # Daha fazla sonuç getir, sonra filtrele
 fetch_k = top_k * 3 if file_filter else top_k

 results = vector_store.similarity_search_with_score(
 query=query,
 k=fetch_k,
 )

 # Dosya türü filtresi uygula
 if file_filter:
 results = [
 (doc, score) for doc, score in results
 if doc.metadata.get("file_type", "") == file_filter
 ][:top_k]

 # Sonuçları logla
 logger.info(
 f"Semantic search: '{query[:50]}...' → {len(results)} sonuç "
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
 """Hibrit arama (BM25 + Semantic) ile en benzer doküman parçalarını bul.

 BM25 (term frequency) ve semantic search sonuçlarını ağırlıklı olarak birleştirir.

 Args:
 vector_store: Vektör deposu nesnesi.
 query: Arama sorgusu.
 chunks: Tüm doküman parçaları (BM25 için gerekli).
 top_k: Döndürülecek sonuç sayısı. None ise config'ten alınır.
 semantic_weight: Semantic arama ağırlığı (0.0-1.0).
 BM25 ağırlığı = 1 - semantic_weight.

 Returns:
 List[Tuple[Document, float]]: (doküman, birleşik_skor) çiftlerinin listesi.
 """
 from rank_bm25 import BM25Okapi

 top_k = top_k or config.top_k

 # BM25 araması
 corpus = [doc.page_content for doc in chunks]
 tokenized_corpus = [doc.lower().split() for doc in corpus]
 bm25 = BM25Okapi(tokenized_corpus)

 tokenized_query = query.lower().split()
 bm25_scores = bm25.get_scores(tokenized_query)

 # BM25 skorlarını normalize et (0-1 arası)
 max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
 bm25_normalized = {i: score / max_bm25 for i, score in enumerate(bm25_scores)}

 # Semantic arama
 semantic_results = vector_store.similarity_search_with_score(
 query=query,
 k=min(len(chunks), top_k * 3),
 )

 # Semantic skorları normalize et (FAISS distance → similarity)
 if semantic_results:
 max_dist = max(score for _, score in semantic_results) or 1
 semantic_map = {}
 for doc, score in semantic_results:
 # FAISS skorunu benzerlik skoruna çevir (düşük mesafe = yüksek benzerlik)
 similarity = 1 - (score / max_dist) if max_dist > 0 else 0
 content_key = doc.page_content[:100]
 semantic_map[content_key] = (doc, similarity)

 # Tüm dokümanlar için hibrit skor hesapla
 combined_scores = []
 for i, chunk in enumerate(chunks):
 content_key = chunk.page_content[:100]
 bm25_score = bm25_normalized.get(i, 0)
 semantic_score = 0

 if content_key in semantic_map:
 _, semantic_score = semantic_map[content_key]

 # Ağırlıklı birleşim
 hybrid_score = (semantic_weight * semantic_score +
 (1 - semantic_weight) * bm25_score)
 combined_scores.append((chunk, hybrid_score))

 # En yüksek skorlu dokümanları sırala
 combined_scores.sort(key=lambda x: x[1], reverse=True)
 results = combined_scores[:top_k]

 logger.info(
 f"Hybrid search: '{query[:50]}...' → {len(results)} sonuç "
 f"(semantic_weight={semantic_weight})"
 )

 return results


def display_search_results(
 results: List[Tuple[Document, float]],
 search_type: str = "semantic",
) -> None:
 """Arama sonuçlarını renkli formatta göster.

 Args:
 results: (doküman, skor) çiftlerinin listesi.
 search_type: Arama tipi ('semantic' veya 'hybrid').
 """
 print(f"\n{Fore.CYAN} Arama Sonuçları ({search_type.capitalize()}){Style.RESET_ALL}")
 print(f"{'─' * 60}")

 if not results:
 print(f" {Fore.YELLOW}Sonuç bulunamadı.{Style.RESET_ALL}")
 return

 for i, (doc, score) in enumerate(results, 1):
 source = doc.metadata.get("source_file", doc.metadata.get("source_short", "Bilinmiyor"))

 # Skor rengini belirle
 if search_type == "semantic":
 # FAISS mesafe skoru: düşük = iyi
 score_color = Fore.GREEN if score < 0.5 else (Fore.YELLOW if score < 1.0 else Fore.RED)
 score_label = f"Mesafe: {score:.4f}"
 else:
 # Hybrid skor: yüksek = iyi
 score_color = Fore.GREEN if score > 0.7 else (Fore.YELLOW if score > 0.3 else Fore.RED)
 score_label = f"Skor: {score:.4f}"

 print(f"\n {Fore.WHITE}[{i}]{Style.RESET_ALL} {score_color}{score_label}{Style.RESET_ALL}")
 print(f" Kaynak: {Fore.BLUE}{source}{Style.RESET_ALL}")

 # İçerik önizlemesi (ilk 200 karakter)
 preview = doc.page_content[:200].replace("\n", " ").strip()
 print(f" {preview}...")

 print(f"\n{'─' * 60}")


def get_retriever(
 vector_store,
 search_type: str = "similarity",
 top_k: Optional[int] = None,
):
 """Vektör deposundan LangChain retriever nesnesi oluştur.

 Args:
 vector_store: Vektör deposu nesnesi.
 search_type: Arama tipi ('similarity' veya 'mmr').
 top_k: Döndürülecek sonuç sayısı.

 Returns:
 LangChain Retriever nesnesi.
 """
 top_k = top_k or config.top_k

 retriever = vector_store.as_retriever(
 search_type=search_type,
 search_kwargs={"k": top_k},
 )

 logger.info(f"Retriever oluşturuldu: type={search_type}, k={top_k}")

 return retriever
