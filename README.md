# RAG-Based Question Answering System

LangChain ve Google Gemini kullanan, dokümanlara dayalı soru-cevap sistemi. Yüklenen belgeleri vektör veritabanında indeksler ve doğal dil sorularını bu belgelerden cevaplar.

## Özellikler

- PDF, TXT, DOCX ve MD formatlarını destekler
- FAISS ve ChromaDB ile vektör indeksleme
- Semantic ve hybrid (BM25 + semantic) arama
- Yapılandırılabilir chunk stratejisi: `recursive` (varsayılan) veya `header` (Markdown)
- JSON tabanlı sorgu önbelleği (TTL destekli)
- Retrieval ve QA kalite değerlendirmesi
- Tüm ayarlar ortam değişkenleriyle yapılandırılabilir
- Streamlit web arayüzü — sohbet, doküman yönetimi, arama gezgini
- Docker & GitHub Actions CI desteği

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

### Web Arayüzü (önerilen)

```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

### Docker

```bash
# .env dosyasını oluştur (GEMINI_API_KEY gerekli)
cp .env.example .env

docker compose up --build
```

Uygulama `http://localhost:8501` adresinde erişilebilir olur. `documents/`, `index/`, `cache/` ve `logs/` dizinleri volume olarak mount edilir — container yeniden başlasa bile veriler korunur.

### Komut Satırı (CLI)

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

İndeksi yeniden oluşturmak için:

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
| `GEMINI_API_KEY` | — | Google Gemini API anahtarı (zorunlu) |
| `LLM_MODEL` | `gemini-2.5-flash` | Cevap üretme modeli |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding modeli |
| `TEMPERATURE` | `0.3` | Üretim sıcaklığı (0.0 – 1.0) |
| `CHUNK_SIZE` | `1000` | Metin parça boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Parçalar arası örtüşme |
| `CHUNK_STRATEGY` | `recursive` | Bölme stratejisi: `recursive` veya `header` |
| `TOP_K` | `5` | Döndürülecek sonuç sayısı |
| `VECTOR_DB` | `faiss` | Vektör DB: `faiss` veya `chroma` |
| `LOG_LEVEL` | `INFO` | Log seviyesi |

### Chunk Stratejileri

| Strateji | Açıklama | Ne zaman kullanılır |
|---|---|---|
| `recursive` | Karakter bazlı özyinelemeli bölme | Her dosya formatı için varsayılan |
| `header` | Markdown `#` `##` `###` başlıklarına göre bölme | `.md` belgeleri için önerilir |

`header` stratejisi seçildiğinde `.md` olmayan dosyalar otomatik olarak `recursive`'e döner.

## Proje Yapısı

```
RAG-Based-Question-Answering-System/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI
├── src/
│   ├── config.py                 # Yapılandırma ve loglama
│   ├── document_loader.py        # Doküman yükleme
│   ├── text_splitter.py          # Metin bölme (recursive + header)
│   ├── embeddings.py             # Vektör embedding ve depolama
│   ├── retriever.py              # Arama ve getirme
│   ├── generator.py              # LLM ile cevap üretme
│   ├── evaluator.py              # Değerlendirme
│   └── cache.py                  # Sorgu önbelleği
├── documents/                    # Kullanıcı belgeleri
├── sample_docs/                  # Örnek belgeler
├── tests/                        # Birim testleri
├── test_questions.json           # Değerlendirme soruları
├── main.py                       # CLI giriş noktası
├── app.py                        # Streamlit web arayüzü
├── Dockerfile                    # Docker image tanımı
├── docker-compose.yml            # Docker Compose yapılandırması
├── .dockerignore
├── .env.example                  # Örnek yapılandırma
└── requirements.txt
```

## Testler

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

CI pipeline her `push` ve `pull_request`'te otomatik olarak testleri çalıştırır.

## Nasıl Çalışıyor

İndeksleme aşamasında belgeler okunur, seçilen chunk stratejisine göre parçalara bölünür ve her parça Gemini embedding modeli kullanılarak vektöre dönüştürülür. Bu vektörler FAISS veya ChromaDB'ye kaydedilir.

Soru sorulduğunda, soru da aynı embedding modeliyle vektöre çevrilir ve vektör veritabanında en benzer parçalar bulunur. Bulunan parçalar bağlam olarak Gemini'ye iletilir ve model yalnızca bu bağlamı kullanarak cevap üretir.

## Lisans

MIT
