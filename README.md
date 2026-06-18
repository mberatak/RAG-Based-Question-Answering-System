# RAG-Based Question Answering System

LangChain ve Google Gemini kullanan, dokümanlara dayalı soru-cevap sistemi. Yüklenen belgeleri vektör veritabanında indeksler ve doğal dil sorularını bu belgelerden cevaplar.

## Özellikler

- PDF, TXT ve DOCX formatlarını destekler
- FAISS ve ChromaDB ile vektör indeksleme
- Semantic ve hybrid (BM25 + semantic) arama
- JSON tabanlı sorgu önbelleği
- Retrieval ve QA kalite değerlendirmesi
- Tüm ayarlar ortam değişkenleriyle yapılandırılabilir

## Kurulum

```bash
git clone https://github.com/mberatak/RAG-Based-Question-Answering-System.git
cd RAG-Based-Question-Answering-System

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

API anahtarını ayarla:

```bash
copy .env.example .env
```

`.env` dosyasını düzenle:

```
GEMINI_API_KEY=your-api-key-here
```

## Kullanım

Belgeleri `documents/` klasörüne koy, ardından indeksle:

```bash
python main.py --ingest
```

Belirli bir dizinden yüklemek için:

```bash
python main.py --ingest --source sample_docs
```

Soru sor:

```bash
python main.py --query "Sorunuz burada"
```

Interaktif mod:

```bash
python main.py --interactive
```

Interaktif moddan çıkmak için `q` veya `çık` yaz. Önbelleği temizlemek için `temizle`.

İndeksi sıfırdan yeniden oluşturmak için:

```bash
python main.py --rebuild
```

Değerlendirme çalıştırmak için:

```bash
python main.py --evaluate
```

## Yapılandırma

Tüm ayarlar `.env` dosyasından veya ortam değişkenleriyle değiştirilebilir:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GEMINI_API_KEY` | - | Google Gemini API anahtarı (zorunlu) |
| `LLM_MODEL` | `gemini-2.5-flash` | Cevap üretme modeli |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding modeli |
| `TEMPERATURE` | `0.3` | Üretim sıcaklığı (0.0 - 1.0) |
| `CHUNK_SIZE` | `1000` | Metin parça boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Parçalar arası örtüşme |
| `TOP_K` | `5` | Döndürülecek sonuç sayısı |
| `VECTOR_DB` | `faiss` | Vektör DB: `faiss` veya `chroma` |
| `LOG_LEVEL` | `INFO` | Log seviyesi |

## Proje Yapısı

```
RAG-Based-Question-Answering-System/
├── src/
│   ├── config.py             # Yapılandırma ve loglama
│   ├── document_loader.py    # Doküman yükleme
│   ├── text_splitter.py      # Metin bölme
│   ├── embeddings.py         # Vektör embedding ve depolama
│   ├── retriever.py          # Arama ve getirme
│   ├── generator.py          # LLM ile cevap üretme
│   ├── evaluator.py          # Değerlendirme
│   └── cache.py              # Sorgu önbelleği
├── documents/                # Kullanıcı belgeleri
├── sample_docs/              # Örnek belgeler
├── tests/                    # Birim testleri
├── test_questions.json       # Değerlendirme soruları
├── main.py                   # CLI giriş noktası
├── .env.example              # Örnek yapılandırma
└── requirements.txt
```

## Testler

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Nasıl Çalışıyor

İndeksleme aşamasında belgeler okunur, parçalara bölünür ve her parça Gemini embedding modeli kullanılarak vektöre dönüştürülür. Bu vektörler FAISS veya ChromaDB'ye kaydedilir.

Soru sorulduğunda, soru da aynı embedding modeliyle vektöre çevrilir ve vektör veritabanında en benzer parçalar bulunur. Bulunan parçalar bağlam olarak Gemini'ye iletilir ve model yalnızca bu bağlamı kullanarak cevap üretir.

## Lisans

MIT
