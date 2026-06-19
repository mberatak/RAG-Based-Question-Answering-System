"""Generator modülü birim testleri."""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestGetLLM:
    """LLM oluşturma testleri."""

    @patch("src.generator.ChatGoogleGenerativeAI")
    def test_get_llm_default_params(self, mock_chat):
        """Varsayılan parametrelerle LLM oluşturulduğunu kontrol et."""
        from src.generator import get_llm

        mock_chat.return_value = MagicMock()
        llm = get_llm()

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["streaming"] is False

    @patch("src.generator.ChatGoogleGenerativeAI")
    def test_get_llm_custom_params(self, mock_chat):
        """Özel parametrelerle LLM oluşturulduğunu kontrol et."""
        from src.generator import get_llm

        mock_chat.return_value = MagicMock()
        get_llm(model="gemini-1.5-pro", temperature=0.7, streaming=True)

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["model"] == "gemini-1.5-pro"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["streaming"] is True


class TestPromptTemplate:
    """Prompt template testleri."""

    def test_prompt_template_contains_turkish(self):
        """Prompt template'in Türkçe talimatlar içerdiğini kontrol et."""
        from src.generator import PROMPT_TEMPLATE

        assert "baglam" in PROMPT_TEMPLATE.lower()
        assert "soru" in PROMPT_TEMPLATE.lower()
        assert "cevap" in PROMPT_TEMPLATE.lower()
        assert "turkce" in PROMPT_TEMPLATE.lower()

    def test_prompt_template_has_variables(self):
        """Prompt template'in gerekli değişkenleri içerdiğini kontrol et."""
        from src.generator import PROMPT_TEMPLATE

        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE


class TestCreateQAChain:
    """QA zinciri oluşturma testleri."""

    @patch("src.generator.RetrievalQA")
    @patch("src.generator.get_retriever")
    @patch("src.generator.ChatGoogleGenerativeAI")
    def test_create_qa_chain_returns_chain(self, mock_chat, mock_retriever, mock_qa):
        """QA zincirinin oluşturulduğunu kontrol et."""
        from src.generator import create_qa_chain

        mock_chat.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_qa.from_chain_type.return_value = MagicMock()
        mock_store = MagicMock()

        chain = create_qa_chain(mock_store)

        assert chain is not None
        mock_qa.from_chain_type.assert_called_once()
