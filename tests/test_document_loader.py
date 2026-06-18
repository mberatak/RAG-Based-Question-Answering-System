"""Document loader modülü birim testleri."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document


class TestDocumentLoader:
    """Doküman yükleme testleri."""

    def test_load_documents_from_sample_docs(self):
        """sample_docs dizinindeki dokümanların yüklendiğini kontrol et."""
        from src.document_loader import load_documents

        sample_dir = str(PROJECT_ROOT / "sample_docs")
        docs = load_documents(sample_dir)

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_documents_metadata(self):
        """Yüklenen dokümanların metadata içerdiğini kontrol et."""
        from src.document_loader import load_documents

        sample_dir = str(PROJECT_ROOT / "sample_docs")
        docs = load_documents(sample_dir)

        for doc in docs:
            assert "source_file" in doc.metadata
            assert "file_type" in doc.metadata

    def test_load_documents_empty_dir(self, tmp_path):
        """Boş dizinde boş liste döndürüldüğünü kontrol et."""
        from src.document_loader import load_documents

        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_load_documents_nonexistent_dir(self):
        """Var olmayan dizinde boş liste döndürüldüğünü kontrol et."""
        from src.document_loader import load_documents

        docs = load_documents("/nonexistent/path")
        assert docs == []

    def test_load_documents_with_file_filter(self):
        """Dosya türü filtresinin çalıştığını kontrol et."""
        from src.document_loader import load_documents

        sample_dir = str(PROJECT_ROOT / "sample_docs")
        docs = load_documents(sample_dir, file_types=[".txt"])

        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata["file_type"] == ".txt"

    def test_load_single_document(self):
        """Tek dosya yüklemenin çalıştığını kontrol et."""
        from src.document_loader import load_single_document

        file_path = str(PROJECT_ROOT / "sample_docs" / "turkiye_sehirleri.txt")
        docs = load_single_document(file_path)

        assert len(docs) > 0
        assert "İstanbul" in docs[0].page_content

    def test_load_single_document_not_found(self):
        """Var olmayan dosya için hata fırlatıldığını kontrol et."""
        from src.document_loader import load_single_document

        with pytest.raises(FileNotFoundError):
            load_single_document("/nonexistent/file.txt")

    def test_load_single_document_unsupported_format(self, tmp_path):
        """Desteklenmeyen formatta hata fırlatıldığını kontrol et."""
        from src.document_loader import load_single_document

        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")

        with pytest.raises(ValueError, match="Desteklenmeyen"):
            load_single_document(str(unsupported))

    def test_turkish_content_encoding(self):
        """Türkçe karakterlerin doğru okunduğunu kontrol et."""
        from src.document_loader import load_single_document

        file_path = str(PROJECT_ROOT / "sample_docs" / "turkiye_sehirleri.txt")
        docs = load_single_document(file_path)

        content = docs[0].page_content
        assert "Türkiye" in content
        assert "İstanbul" in content
        assert "Ankara" in content
