"""Sorgu onbellekleme modulu.

Ayni sorularin tekrar sorulmasi durumunda API cagrisi yapmadan
onbellekten cevap dondurur. JSON tabanli disk onbellegi kullanir.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from colorama import Fore, Style

from src.config import config, logger


class QueryCache:
    """JSON tabanli sorgu onbellegi.

    Attributes:
        cache_file: Onbellek dosyasi yolu.
        ttl_hours: Onbellek gecerlilik suresi (saat).
        cache: Bellek ici onbellek sozlugu.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ):
        """Onbellegi baslat.

        Args:
            cache_dir: Onbellek dizini. None ise config'ten alinir.
            ttl_hours: Gecerlilik suresi (saat). None ise config'ten alinir.
        """
        cache_dir = Path(cache_dir) if cache_dir else config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = cache_dir / "query_cache.json"
        self.ttl_hours = ttl_hours or config.cache_ttl_hours
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Onbellegi diskten yukle.

        Returns:
            Dict[str, Any]: Onbellek sozlugu.
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                logger.debug(f"Onbellek yuklendi: {len(cache)} giris")
                return cache
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Onbellek yuklenemedi: {e}")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Onbellegi diske kaydet."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Onbellek kaydedildi: {len(self.cache)} giris")
        except IOError as e:
            logger.error(f"Onbellek kaydedilemedi: {e}")

    @staticmethod
    def _hash_query(query: str) -> str:
        """Sorgu icin benzersiz hash olustur.

        Args:
            query: Sorgu metni.

        Returns:
            str: SHA-256 hash degeri.
        """
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Onbellekten sorgu sonucunu getir.

        Args:
            query: Sorgu metni.

        Returns:
            Optional[Dict[str, Any]]: Onbellekteki cevap veya None.
        """
        query_hash = self._hash_query(query)

        if query_hash not in self.cache:
            return None

        entry = self.cache[query_hash]

        # TTL kontrolu
        cached_time = datetime.fromisoformat(entry["timestamp"])
        if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
            del self.cache[query_hash]
            self._save_cache()
            logger.debug(f"Onbellek suresi dolmus: '{query[:30]}...'")
            return None

        logger.info(f"Onbellekten donduruldu: '{query[:30]}...'")
        return entry["data"]

    def set(self, query: str, data: Dict[str, Any]) -> None:
        """Sorgu sonucunu onbellege kaydet.

        Args:
            query: Sorgu metni.
            data: Kaydedilecek cevap verisi.
        """
        query_hash = self._hash_query(query)

        # Source documents serializable hale getir
        serializable_data = {
            "answer": data.get("answer", ""),
            "total_tokens": data.get("total_tokens", 0),
            "total_cost": data.get("total_cost", 0.0),
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source_file", ""),
                }
                for doc in data.get("source_documents", [])
            ],
        }

        self.cache[query_hash] = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "data": serializable_data,
        }

        self._save_cache()
        logger.debug(f"Onbellege kaydedildi: '{query[:30]}...'")

    def clear(self) -> int:
        """Tum onbellegi temizle.

        Returns:
            int: Silinen giris sayisi.
        """
        count = len(self.cache)
        self.cache = {}
        self._save_cache()
        logger.info(f"Onbellek temizlendi: {count} giris")
        return count

    def stats(self) -> Dict[str, Any]:
        """Onbellek istatistiklerini dondur.

        Returns:
            Dict[str, Any]: Istatistik bilgileri.
        """
        valid_count = 0
        expired_count = 0

        for entry in self.cache.values():
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                expired_count += 1
            else:
                valid_count += 1

        stats = {
            "total_entries": len(self.cache),
            "valid_entries": valid_count,
            "expired_entries": expired_count,
            "ttl_hours": self.ttl_hours,
            "cache_file": str(self.cache_file),
        }

        return stats
