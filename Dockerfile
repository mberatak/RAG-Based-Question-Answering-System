# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Sistem bağımlılıkları (PDF işleme için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Sistem runtime bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini builder'dan kopyala
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Uygulama kodunu kopyala
COPY src/ ./src/
COPY app.py main.py ./
COPY .env.example ./

# Gerekli dizinleri oluştur
RUN mkdir -p documents index cache logs

# Non-root kullanıcı
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
