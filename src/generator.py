"""Cevap üretme modülü.

LangChain RetrievalQA zinciri ile Google Gemini LLM kullanarak soru-cevap üretir.
Prompt template, kaynak doküman gösterimi içerir.
"""

from typing import Dict, List, Optional, Any

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from colorama import Fore, Style

from src.config import config, logger
from src.retriever import get_retriever

# Türkçe prompt template
PROMPT_TEMPLATE = """Sen bir doküman asistanısın. Sadece verilen bağlamı kullanarak soruyu cevapla.
Cevabını Türkçe ver. Emin değilsen veya bağlamda bilgi yoksa 'Bu konuda bilgim yok' de.
Cevabını kısa ve öz tut, gereksiz bilgi ekleme.

Bağlam:
{context}

Soru: {question}

Cevap:"""


def get_llm(
 model: Optional[str] = None,
 temperature: Optional[float] = None,
 streaming: bool = False,
) -> ChatGoogleGenerativeAI:
 """Google Gemini LLM modelini oluştur ve döndür.

 Args:
 model: Kullanılacak model adı. None ise config'ten alınır.
 temperature: Sıcaklık değeri. None ise config'ten alınır.
 streaming: Stream modu aktif mi.

 Returns:
 ChatGoogleGenerativeAI: Yapılandırılmış LLM nesnesi.
 """
 return ChatGoogleGenerativeAI(
 model=model or config.llm_model,
 temperature=temperature if temperature is not None else config.temperature,
 google_api_key=config.gemini_api_key,
 streaming=streaming,
 )


def create_qa_chain(
 vector_store,
 model: Optional[str] = None,
 temperature: Optional[float] = None,
 top_k: Optional[int] = None,
) -> RetrievalQA:
 """RetrievalQA zinciri oluştur.

 Args:
 vector_store: Vektör deposu nesnesi.
 model: LLM model adı. None ise config'ten alınır.
 temperature: LLM sıcaklık değeri. None ise config'ten alınır.
 top_k: Döndürülecek doküman sayısı. None ise config'ten alınır.

 Returns:
 RetrievalQA: Yapılandırılmış QA zinciri.
 """
 llm = get_llm(model=model, temperature=temperature)
 retriever = get_retriever(vector_store, top_k=top_k)

 prompt = PromptTemplate(
 template=PROMPT_TEMPLATE,
 input_variables=["context", "question"],
 )

 qa_chain = RetrievalQA.from_chain_type(
 llm=llm,
 chain_type="stuff",
 retriever=retriever,
 return_source_documents=True,
 chain_type_kwargs={"prompt": prompt},
 )

 logger.info(
 f"QA zinciri oluşturuldu: model={model or config.llm_model}, "
 f"temp={temperature or config.temperature}"
 )

 return qa_chain


def ask_question(
 qa_chain: RetrievalQA,
 question: str,
 show_sources: bool = True,
 show_tokens: bool = True,
) -> Dict[str, Any]:
 """Soru sor ve cevap al.

 Args:
 qa_chain: RetrievalQA zinciri.
 question: Sorulacak soru.
 show_sources: Kaynak dokümanları göster.
 show_tokens: Token kullanımını göster (Gemini'de tahmini gösterilir).

 Returns:
 Dict[str, Any]: Cevap, kaynaklar ve token bilgilerini içeren sözlük.
 - 'answer': str - Üretilen cevap.
 - 'source_documents': List[Document] - Kaynak dokümanlar.
 - 'total_tokens': int - Toplam token kullanımı (tahmini).
 - 'total_cost': float - Toplam maliyet (0.0 — Gemini ücretsiz kota).

 Example:
 >>> result = ask_question(chain, "Türkiye'nin başkenti nedir?")
 >>> print(result['answer'])
 Ankara
 """
 print(f"\n{Fore.CYAN} Soru:{Style.RESET_ALL} {question}")
 print(f"{'─' * 60}")

 response = qa_chain.invoke({"query": question})

 answer = response.get("result", "Cevap üretilemedi.")
 source_docs = response.get("source_documents", [])

 # Token bilgisi — Gemini usage_metadata üzerinden
 usage = getattr(response.get("__end__", None), "usage_metadata", None)
 prompt_tokens = 0
 completion_tokens = 0
 if usage:
 prompt_tokens = getattr(usage, "prompt_token_count", 0)
 completion_tokens = getattr(usage, "candidates_token_count", 0)
 total_tokens = prompt_tokens + completion_tokens

 # Cevabı göster
 print(f"\n{Fore.GREEN} Cevap:{Style.RESET_ALL}")
 print(f" {answer}")

 # Kaynak dokümanları göster
 if show_sources and source_docs:
 print(f"\n{Fore.CYAN} Kaynaklar:{Style.RESET_ALL}")
 seen_sources = set()
 for i, doc in enumerate(source_docs, 1):
 source = doc.metadata.get(
 "source_file",
 doc.metadata.get("source", "Bilinmiyor"),
 )
 source_key = f"{source}"
 if source_key in seen_sources:
 continue
 seen_sources.add(source_key)

 preview = doc.page_content[:100].replace("\n", " ").strip()
 print(f" [{i}] {Fore.BLUE}{source}{Style.RESET_ALL}")
 print(f" {preview}...")

 # Token bilgisini göster
 if show_tokens and total_tokens > 0:
 print(f"\n{Fore.YELLOW} Token Kullanımı:{Style.RESET_ALL}")
 print(f" Prompt: {prompt_tokens:,} token")
 print(f" Cevap: {completion_tokens:,} token")
 print(f" Toplam: {total_tokens:,} token")

 print(f"{'─' * 60}")

 result = {
 "answer": answer,
 "source_documents": source_docs,
 "total_tokens": total_tokens,
 "prompt_tokens": prompt_tokens,
 "completion_tokens": completion_tokens,
 "total_cost": 0.0, # Gemini ücretsiz kota — maliyet takibi yapılmıyor
 }

 logger.info(
 f"Soru cevaplandı: '{question[:50]}...' → {total_tokens} token"
 )

 return result


def ask_question_with_streaming(
 vector_store,
 question: str,
 model: Optional[str] = None,
) -> str:
 """Stream modunda soru sor ve cevabı gerçek zamanlı olarak göster.

 Args:
 vector_store: Vektör deposu nesnesi.
 question: Sorulacak soru.
 model: LLM model adı. None ise config'ten alınır.

 Returns:
 str: Üretilen tam cevap.
 """
 from langchain_google_genai import ChatGoogleGenerativeAI
 from langchain_core.prompts import PromptTemplate
 from langchain_core.output_parsers import StrOutputParser
 from langchain_core.runnables import RunnablePassthrough

 llm = ChatGoogleGenerativeAI(
 model=model or config.llm_model,
 temperature=config.temperature,
 google_api_key=config.gemini_api_key,
 streaming=True,
 )

 retriever = get_retriever(vector_store)

 prompt = PromptTemplate(
 template=PROMPT_TEMPLATE,
 input_variables=["context", "question"],
 )

 def format_docs(docs):
 return "\n\n".join(doc.page_content for doc in docs)

 chain = (
 {"context": retriever | format_docs, "question": RunnablePassthrough()}
 | prompt
 | llm
 | StrOutputParser()
 )

 print(f"\n{Fore.CYAN} Soru:{Style.RESET_ALL} {question}")
 print(f"{Fore.GREEN} Cevap:{Style.RESET_ALL} ", end="", flush=True)

 full_response = ""
 for chunk in chain.stream(question):
 print(chunk, end="", flush=True)
 full_response += chunk

 print(f"\n{'─' * 60}")

 return full_response
