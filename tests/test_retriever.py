"""Retriever modülü birim testleri."""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from src.retriever import display_search_results, get_retriever


class TestDisplaySearchResults:
    """Arama sonuçları gösterimi testleri."""

    def test_display_empty_results(self, capsys):
        """Boş sonuçların düzgün gösterildiğini kontrol et."""
        display_search_results([], search_type="semantic")
        captured = capsys.readouterr()
        assert "Sonuc bulunamadi" in captured.out

    def test_display_semantic_results(self, capsys):
        """Semantic arama sonuçlarının gösterildiğini kontrol et."""
        results = [
            (
                Document(
                    page_content="Test doküman içeriği",
                    metadata={"source_file": "test.txt"},
                ),
                0.3,
            ),
        ]
        display_search_results(results, search_type="semantic")
        captured = capsys.readouterr()
        assert "Mesafe" in captured.out
        assert "test.txt" in captured.out

    def test_display_hybrid_results(self, capsys):
        """Hybrid arama sonuçlarının gösterildiğini kontrol et."""
        results = [
            (
                Document(
                    page_content="Hybrid test içeriği",
                    metadata={"source_file": "hybrid.txt"},
                ),
                0.85,
            ),
        ]
        display_search_results(results, search_type="hybrid")
        captured = capsys.readouterr()
        assert "Skor" in captured.out


class TestGetRetriever:
    """Retriever oluşturma testleri."""

    def test_get_retriever_returns_object(self):
        """Retriever nesnesinin döndürüldüğünü kontrol et."""
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        retriever = get_retriever(mock_store, top_k=3)

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        assert retriever == mock_retriever

    def test_get_retriever_mmr_search(self):
        """MMR arama tipinin çalıştığını kontrol et."""
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()

        get_retriever(mock_store, search_type="mmr", top_k=5)

        mock_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 5},
        )


class TestSemanticSearch:
    """Semantic search testleri (mock ile)."""

    def test_semantic_search_basic(self):
        """Basit semantic aramanın çalıştığını kontrol et."""
        from src.retriever import semantic_search

        mock_store = MagicMock()
        mock_doc = Document(
            page_content="Ankara Türkiye'nin başkentidir.",
            metadata={"source_file": "test.txt", "file_type": ".txt"},
        )
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.2)]

        results = semantic_search(mock_store, "Başkent neresi?", top_k=3)

        assert len(results) == 1
        assert results[0][0].page_content == "Ankara Türkiye'nin başkentidir."

    def test_semantic_search_with_filter(self):
        """Dosya filtresiyle aramanın çalıştığını kontrol et."""
        from src.retriever import semantic_search

        mock_store = MagicMock()
        doc_pdf = Document(
            page_content="PDF içerik",
            metadata={"source_file": "doc.pdf", "file_type": ".pdf"},
        )
        doc_txt = Document(
            page_content="TXT içerik",
            metadata={"source_file": "doc.txt", "file_type": ".txt"},
        )
        mock_store.similarity_search_with_score.return_value = [
            (doc_pdf, 0.1),
            (doc_txt, 0.2),
        ]

        results = semantic_search(
            mock_store, "test", top_k=5, file_filter=".txt"
        )

        assert len(results) == 1
        assert results[0][0].metadata["file_type"] == ".txt"
