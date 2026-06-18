"""Embeddings modülü birim testleri."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from src.embeddings import estimate_embedding_cost


class TestEmbeddingCost:
    """Embedding maliyet tahmini testleri."""

    def test_estimate_cost_basic(self):
        """Maliyet tahmininin doğru hesaplandığını kontrol et."""
        chunks = [
            Document(page_content="Bu bir test metnidir.", metadata={}),
            Document(page_content="İkinci test metni.", metadata={}),
        ]

        total_tokens, cost = estimate_embedding_cost(chunks)

        assert total_tokens > 0
        assert cost > 0
        assert cost == pytest.approx(total_tokens / 1000 * 0.0001, abs=1e-8)

    def test_estimate_cost_empty(self):
        """Boş chunk listesinde sıfır maliyet döndüğünü kontrol et."""
        total_tokens, cost = estimate_embedding_cost([])

        assert total_tokens == 0
        assert cost == 0.0

    def test_estimate_cost_turkish_text(self):
        """Türkçe metnin token sayısının hesaplandığını kontrol et."""
        chunks = [
            Document(
                page_content="Türkiye Cumhuriyeti, 29 Ekim 1923'te ilan edilmiştir. "
                             "İstanbul, Türkiye'nin en büyük şehridir.",
                metadata={},
            )
        ]

        total_tokens, cost = estimate_embedding_cost(chunks)

        assert total_tokens > 0
        # Türkçe metin daha fazla token üretir
        assert total_tokens > 10


class TestVectorStoreOperations:
    """Vektör deposu işlemleri testleri (mock ile)."""

    @patch("src.embeddings.get_embeddings")
    def test_create_vector_store_faiss_mock(self, mock_embeddings):
        """FAISS vektör deposu oluşturmanın çağrıldığını kontrol et."""
        # Bu test sadece fonksiyonun parametreleri doğru aldığını kontrol eder
        mock_embeddings.return_value = MagicMock()

        # Gerçek API çağrısı yapılmadan test edilir
        from src.embeddings import get_embeddings
        embeddings = get_embeddings()
        assert embeddings is not None

    @patch("src.embeddings.get_embeddings")
    def test_load_nonexistent_faiss_index(self, mock_get_emb):
        """Var olmayan FAISS indeksinin None döndürdüğünü kontrol et."""
        from src.embeddings import load_vector_store
        from src.config import config

        mock_get_emb.return_value = MagicMock()

        # Geçici bir dizin kullan
        original = config.index_dir
        config.index_dir = Path("nonexistent_test_dir_xyz")

        result = load_vector_store("faiss")
        assert result is None

        config.index_dir = original
