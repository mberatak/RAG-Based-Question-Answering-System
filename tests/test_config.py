"""Config modülü birim testleri."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, setup_logging


class TestConfig:
    """Config sınıfı testleri."""

    def test_default_values(self):
        """Varsayılan değerlerin doğru atandığını kontrol et."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
            assert cfg.embedding_model == "gemini-embedding-001"
            assert cfg.llm_model == "gemini-2.5-flash"
            assert cfg.temperature == 0.3
            assert cfg.chunk_size == 1000
            assert cfg.chunk_overlap == 200
            assert cfg.top_k == 5
            assert cfg.vector_db == "faiss"
            assert cfg.log_level == "INFO"

    def test_env_override(self):
        """Ortam değişkenlerinin ayarları geçersiz kıldığını kontrol et."""
        env_vars = {
            "CHUNK_SIZE": "500",
            "CHUNK_OVERLAP": "100",
            "TOP_K": "10",
            "LLM_MODEL": "gpt-4",
            "TEMPERATURE": "0.7",
            "VECTOR_DB": "chroma",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_vars):
            cfg = Config()
            assert cfg.chunk_size == 500
            assert cfg.chunk_overlap == 100
            assert cfg.top_k == 10
            assert cfg.llm_model == "gpt-4"
            assert cfg.temperature == 0.7
            assert cfg.vector_db == "chroma"
            assert cfg.log_level == "DEBUG"

    def test_validate_no_api_key(self):
        """API anahtarı yoksa hata fırlatıldığını kontrol et."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            cfg = Config()
            cfg.gemini_api_key = ""
            with pytest.raises(ValueError):
                cfg.validate()

    def test_validate_invalid_vector_db(self):
        """Geçersiz vektör DB seçiminde hata fırlatıldığını kontrol et."""
        cfg = Config()
        cfg.openai_api_key = "sk-test-key-12345"
        cfg.vector_db = "invalid_db"
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_invalid_chunk_size(self):
        """Geçersiz chunk_size değerinde hata fırlatıldığını kontrol et."""
        cfg = Config()
        cfg.openai_api_key = "sk-test-key-12345"
        cfg.chunk_size = 0
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_success(self):
        """Geçerli ayarlarla doğrulamanın başarılı olduğunu kontrol et."""
        cfg = Config()
        cfg.gemini_api_key = "AIzaSy..."
        assert cfg.validate() is True

    def test_directories_created(self, tmp_path):
        """Gerekli dizinlerin oluşturulduğunu kontrol et."""
        cfg = Config()
        cfg.documents_dir = tmp_path / "docs"
        cfg.index_dir = tmp_path / "idx"
        cfg.cache_dir = tmp_path / "cache"
        cfg.logs_dir = tmp_path / "logs"
        cfg.__post_init__()

        assert cfg.documents_dir.exists()
        assert cfg.index_dir.exists()
        assert cfg.cache_dir.exists()
        assert cfg.logs_dir.exists()


class TestLogging:
    """Logging yapılandırması testleri."""

    def test_setup_logging(self):
        """Logger'ın düzgün yapılandırıldığını kontrol et."""
        cfg = Config()
        log = setup_logging(cfg)
        assert log.name == "rag_system"
        assert len(log.handlers) == 2  # file + console
