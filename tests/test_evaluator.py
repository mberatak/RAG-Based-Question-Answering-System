"""Evaluator modülü birim testleri."""

import json
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import load_test_questions


class TestLoadTestQuestions:
    """Test soruları yükleme testleri."""

    def test_load_existing_questions(self):
        """Mevcut test sorularının yüklendiğini kontrol et."""
        questions = load_test_questions()

        assert len(questions) >= 5
        for q in questions:
            assert "soru" in q
            assert "beklenen_cevap" in q
            assert "ilgili_dosya" in q

    def test_load_nonexistent_file(self):
        """Var olmayan dosyada boş liste döndürüldüğünü kontrol et."""
        questions = load_test_questions("/nonexistent/path.json")
        assert questions == []

    def test_questions_have_turkish_content(self):
        """Soruların Türkçe içerik barındırdığını kontrol et."""
        questions = load_test_questions()

        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        has_turkish = False
        for q in questions:
            if any(c in q["soru"] for c in turkish_chars):
                has_turkish = True
                break

        assert has_turkish, "Test sorularında Türkçe karakterler bulunmalı"

    def test_custom_file_path(self, tmp_path):
        """Özel dosya yolundan yüklemenin çalıştığını kontrol et."""
        test_data = [
            {
                "soru": "Test sorusu",
                "beklenen_cevap": "Test cevap",
                "ilgili_dosya": "test.txt",
            }
        ]
        filepath = tmp_path / "test_q.json"
        filepath.write_text(json.dumps(test_data, ensure_ascii=False), encoding="utf-8")

        questions = load_test_questions(str(filepath))

        assert len(questions) == 1
        assert questions[0]["soru"] == "Test sorusu"


class TestCacheModule:
    """Cache modülü testleri."""

    def test_cache_set_and_get(self, tmp_path):
        """Önbelleğe yazma ve okuma işleminin çalıştığını kontrol et."""
        from src.cache import QueryCache
        from langchain_core.documents import Document

        cache = QueryCache(cache_dir=str(tmp_path), ttl_hours=1)

        data = {
            "answer": "Ankara",
            "total_tokens": 100,
            "total_cost": 0.001,
            "source_documents": [
                Document(
                    page_content="Ankara başkenttir.",
                    metadata={"source_file": "test.txt"},
                )
            ],
        }

        cache.set("Başkent neresi?", data)
        result = cache.get("Başkent neresi?")

        assert result is not None
        assert result["answer"] == "Ankara"

    def test_cache_miss(self, tmp_path):
        """Önbellekte olmayan sorgunun None döndürdüğünü kontrol et."""
        from src.cache import QueryCache

        cache = QueryCache(cache_dir=str(tmp_path))
        result = cache.get("Bu soru önbellekte yok")

        assert result is None

    def test_cache_clear(self, tmp_path):
        """Önbellek temizlemenin çalıştığını kontrol et."""
        from src.cache import QueryCache

        cache = QueryCache(cache_dir=str(tmp_path))
        cache.set("test", {"answer": "cevap", "total_tokens": 0, "total_cost": 0, "source_documents": []})

        count = cache.clear()
        assert count == 1

        result = cache.get("test")
        assert result is None

    def test_cache_stats(self, tmp_path):
        """Önbellek istatistiklerinin doğru döndürüldüğünü kontrol et."""
        from src.cache import QueryCache

        cache = QueryCache(cache_dir=str(tmp_path))
        cache.set("q1", {"answer": "a1", "total_tokens": 0, "total_cost": 0, "source_documents": []})
        cache.set("q2", {"answer": "a2", "total_tokens": 0, "total_cost": 0, "source_documents": []})

        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2

    def test_cache_case_insensitive(self, tmp_path):
        """Önbelleğin büyük/küçük harf duyarsız olduğunu kontrol et."""
        from src.cache import QueryCache

        cache = QueryCache(cache_dir=str(tmp_path))
        cache.set("Test Sorusu", {"answer": "cevap", "total_tokens": 0, "total_cost": 0, "source_documents": []})

        result = cache.get("test sorusu")
        assert result is not None
        assert result["answer"] == "cevap"
