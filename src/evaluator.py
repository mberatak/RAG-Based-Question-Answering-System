"""Değerlendirme metrikleri modülü.

Test soruları üzerinde retrieval accuracy, precision ve recall hesaplar.
Sonuçları tablolu formatta gösterir ve JSON dosyasına kaydeder.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from colorama import Fore, Style

from src.config import config, logger, PROJECT_ROOT
from src.retriever import semantic_search
from src.generator import create_qa_chain, ask_question


def load_test_questions(
    filepath: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Test sorularını JSON dosyasından yükle.

    Args:
        filepath: Test soruları dosya yolu. None ise proje kökünden alınır.

    Returns:
        List[Dict[str, str]]: Test soruları listesi.
        Her soru: {'soru': str, 'beklenen_cevap': str, 'ilgili_dosya': str}
    """
    filepath = filepath or str(PROJECT_ROOT / "test_questions.json")

    if not Path(filepath).exists():
        logger.warning(f"Test soruları dosyası bulunamadı: {filepath}")
        print(f"{Fore.YELLOW} Test soruları dosyası bulunamadı: {filepath}{Style.RESET_ALL}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"{len(questions)} test sorusu yüklendi: {filepath}")
    return questions


def evaluate_retrieval(
    vector_store,
    test_questions: List[Dict[str, str]],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieval doğruluğunu değerlendir.

    Her test sorusu için getirilen dokümanların doğru kaynaktan gelip
    gelmediğini kontrol eder.

    Args:
        vector_store: Vektör deposu nesnesi.
        test_questions: Test soruları listesi.
        top_k: Değerlendirme için top-k değeri.

    Returns:
        Dict[str, Any]: Değerlendirme sonuçları.
    """
    top_k = top_k or config.top_k

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f" Retrieval Değerlendirmesi Başlıyor...")
    print(f" Soru sayısı: {len(test_questions)} | Top-K: {top_k}")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    correct_retrievals = 0
    total_questions = len(test_questions)
    results = []

    for i, q in enumerate(test_questions, 1):
        soru = q["soru"]
        expected_file = q.get("ilgili_dosya", "")

        # Semantic search yap
        search_results = semantic_search(
            vector_store=vector_store,
            query=soru,
            top_k=top_k,
        )

        # Getirilen kaynak dosyaları kontrol et
        retrieved_sources = set()
        for doc, score in search_results:
            source = doc.metadata.get(
                "source_file",
                doc.metadata.get("source", ""),
            )
            # Tam yoldan dosya adını çıkar
            source_name = Path(source).name if source else ""
            retrieved_sources.add(source_name)

        # Doğru dosya getirildi mi?
        is_correct = expected_file in retrieved_sources if expected_file else True
        if is_correct:
            correct_retrievals += 1

        status = f"{Fore.GREEN}{Style.RESET_ALL}" if is_correct else f"{Fore.RED}{Style.RESET_ALL}"
        print(f" {status} [{i}/{total_questions}] {soru}")
        if not is_correct:
            print(f" Beklenen: {expected_file}")
            print(f" Getirilen: {', '.join(retrieved_sources)}")

        results.append({
            "soru": soru,
            "beklenen_dosya": expected_file,
            "getirilen_dosyalar": list(retrieved_sources),
            "dogru": is_correct,
        })

    # Metrikleri hesapla
    accuracy = correct_retrievals / total_questions if total_questions > 0 else 0

    print(f"\n{Fore.CYAN}{'-' * 60}")
    print(f" Retrieval Sonuçları")
    print(f"{'-' * 60}{Style.RESET_ALL}")
    print(f" Doğru: {Fore.GREEN}{correct_retrievals}{Style.RESET_ALL}/{total_questions}")
    print(f" Accuracy: {Fore.GREEN}{accuracy:.1%}{Style.RESET_ALL}")
    print()

    return {
        "accuracy": accuracy,
        "correct": correct_retrievals,
        "total": total_questions,
        "details": results,
    }


def evaluate_qa(
    vector_store,
    test_questions: List[Dict[str, str]],
) -> Dict[str, Any]:
    """QA (soru-cevap) kalitesini değerlendir.

    Her test sorusu için cevap üretir ve beklenen cevapla karşılaştırır.

    Args:
        vector_store: Vektör deposu nesnesi.
        test_questions: Test soruları listesi.

    Returns:
        Dict[str, Any]: Değerlendirme sonuçları.
    """
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f" QA Değerlendirmesi Başlıyor...")
    print(f" Soru sayısı: {len(test_questions)}")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    qa_chain = create_qa_chain(vector_store)

    correct_answers = 0
    total_questions = len(test_questions)
    total_tokens = 0
    total_cost = 0.0
    results = []

    for i, q in enumerate(test_questions, 1):
        soru = q["soru"]
        expected = q.get("beklenen_cevap", "")

        print(f"\n [{i}/{total_questions}] {Fore.WHITE}{soru}{Style.RESET_ALL}")

        # Cevap üret
        result = ask_question(
            qa_chain=qa_chain,
            question=soru,
            show_sources=False,
            show_tokens=False,
        )

        answer = result["answer"]
        total_tokens += result["total_tokens"]
        total_cost += result["total_cost"]

        # Basit eşleşme kontrolü (beklenen cevap, üretilen cevapta geçiyor mu)
        is_correct = expected.lower() in answer.lower() if expected else True
        if is_correct:
            correct_answers += 1

        status = f"{Fore.GREEN}{Style.RESET_ALL}" if is_correct else f"{Fore.RED}{Style.RESET_ALL}"
        print(f" {status} Cevap: {answer[:100]}")
        if not is_correct:
            print(f" {Fore.YELLOW}Beklenen: {expected}{Style.RESET_ALL}")

        results.append({
            "soru": soru,
            "beklenen_cevap": expected,
            "uretilen_cevap": answer,
            "dogru": is_correct,
            "tokens": result["total_tokens"],
        })

    # Metrikleri hesapla
    accuracy = correct_answers / total_questions if total_questions > 0 else 0

    print(f"\n{Fore.CYAN}{'-' * 60}")
    print(f" QA Değerlendirme Sonuçları")
    print(f"{'-' * 60}{Style.RESET_ALL}")
    print(f" Doğru: {Fore.GREEN}{correct_answers}{Style.RESET_ALL}/{total_questions}")
    print(f" Accuracy: {Fore.GREEN}{accuracy:.1%}{Style.RESET_ALL}")
    print(f" Toplam Token: {Fore.YELLOW}{total_tokens:,}{Style.RESET_ALL}")
    print(f" Toplam Maliyet: {Fore.YELLOW}${total_cost:.6f}{Style.RESET_ALL}")
    print()

    return {
        "accuracy": accuracy,
        "correct": correct_answers,
        "total": total_questions,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "details": results,
    }


def run_full_evaluation(
    vector_store,
    test_questions: Optional[List[Dict[str, str]]] = None,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Tam değerlendirme çalıştır (retrieval + QA).

    Args:
        vector_store: Vektör deposu nesnesi.
        test_questions: Test soruları. None ise dosyadan yüklenir.
        save_results: Sonuçları dosyaya kaydet.

    Returns:
        Dict[str, Any]: Tüm değerlendirme sonuçları.
    """
    if test_questions is None:
        test_questions = load_test_questions()

    if not test_questions:
        print(f"{Fore.RED} Test soruları bulunamadı!{Style.RESET_ALL}")
        return {}

    # Retrieval değerlendirmesi
    retrieval_results = evaluate_retrieval(vector_store, test_questions)

    # QA değerlendirmesi
    qa_results = evaluate_qa(vector_store, test_questions)

    # Sonuçları birleştir
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "top_k": config.top_k,
        },
        "retrieval": retrieval_results,
        "qa": qa_results,
    }

    # Dosyaya kaydet
    if save_results:
        output_path = config.logs_dir / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN} Sonuçlar kaydedildi: {output_path}{Style.RESET_ALL}")
        logger.info(f"Değerlendirme sonuçları kaydedildi: {output_path}")

    return evaluation
