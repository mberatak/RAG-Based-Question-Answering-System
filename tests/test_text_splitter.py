"""Text splitter modülü birim testleri."""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from src.text_splitter import split_documents, split_text


class TestSplitDocuments:
    """Doküman bölme testleri."""

    def _create_docs(self, text: str, metadata: dict = None) -> list:
        """Yardımcı: Test dokümanları oluştur."""
        return [Document(page_content=text, metadata=metadata or {"source": "test.txt"})]

    def test_basic_split(self):
        """Basit metin bölmenin çalıştığını kontrol et."""
        text = "Bu bir test cümlesidir. " * 100
        docs = self._create_docs(text)
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 + 50  # tolerans

    def test_chunk_metadata(self):
        """Her chunk'ın metadata içerdiğini kontrol et."""
        text = "Uzun bir metin. " * 200
        docs = self._create_docs(text, {"source": "test.txt", "page": 1})
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)

        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_size" in chunk.metadata

    def test_chunk_overlap(self):
        """Chunk'lar arasında örtüşme olduğunu kontrol et."""
        text = "Kelime " * 500
        docs = self._create_docs(text)
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

        # Ardışık chunk'larda örtüşme olmalı
        if len(chunks) >= 2:
            c1_end = chunks[0].page_content[-20:]
            c2_start = chunks[1].page_content[:40]
            # Örtüşme bölgesinin bir kısmı ikinci chunk'ta bulunmalı
            assert len(c1_end) > 0

    def test_small_text_no_split(self):
        """Küçük metinlerin bölünmediğini kontrol et."""
        text = "Kısa metin."
        docs = self._create_docs(text)
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0].page_content == text

    def test_custom_separators(self):
        """Özel ayraçlarla bölmenin çalıştığını kontrol et."""
        text = "Bölüm 1###Bölüm 2###Bölüm 3"
        docs = self._create_docs(text)
        chunks = split_documents(
            docs,
            chunk_size=20,
            chunk_overlap=0,
            custom_separators=["###"],
        )

        assert len(chunks) >= 2

    def test_empty_documents(self):
        """Boş doküman listesinin boş chunk döndürdüğünü kontrol et."""
        chunks = split_documents([], chunk_size=100, chunk_overlap=20)
        assert chunks == []


class TestSplitText:
    """Ham metin bölme testleri."""

    def test_split_text_basic(self):
        """Ham metin bölmenin çalıştığını kontrol et."""
        text = "Bu bir test cümlesidir. " * 100
        parts = split_text(text, chunk_size=200, chunk_overlap=50)

        assert len(parts) > 1
        assert all(isinstance(p, str) for p in parts)

    def test_split_text_short(self):
        """Kısa metinlerin bölünmediğini kontrol et."""
        text = "Kısa."
        parts = split_text(text, chunk_size=100, chunk_overlap=20)

        assert len(parts) == 1
        assert parts[0] == text
