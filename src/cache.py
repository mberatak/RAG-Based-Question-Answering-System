"""Sorgu önbellekleme modülü.

Aynı soruların tekrar sorulması durumunda API çağrısı yapmadan
önbellekten cevap döndürür. JSON tabanlı disk önbelleği kullanır.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from colorama import Fore, Style

from src.config import config, logger


class QueryCache:
 """JSON tabanlı sorgu önbelleği.

 Attributes:
 cache_file: Önbellek dosyası yolu.
 ttl_hours: Önbellek geçerlilik süresi (saat).
 cache: Bellek içi önbellek sözlüğü.
 """

 def __init__(
 self,
 cache_dir: Optional[str] = None,
 ttl_hours: Optional[int] = None,
 ):
 """Önbelleği başlat.

 Args:
 cache_dir: Önbellek dizini. None ise config'ten alınır.
 ttl_hours: Geçerlilik süresi (saat). None ise config'ten alınır.
 """
 cache_dir = Path(cache_dir) if cache_dir else config.cache_dir
 cache_dir.mkdir(parents=True, exist_ok=True)

 self.cache_file = cache_dir / "query_cache.json"
 self.ttl_hours = ttl_hours or config.cache_ttl_hours
 self.cache: Dict[str, Any] = self._load_cache()

 def _load_cache(self) -> Dict[str, Any]:
 """Önbelleği diskten yükle.

 Returns:
 Dict[str, Any]: Önbellek sözlüğü.
 """
 if self.cache_file.exists():
 try:
 with open(self.cache_file, "r", encoding="utf-8") as f:
 cache = json.load(f)
 logger.debug(f"Önbellek yüklendi: {len(cache)} giriş")
 return cache
 except (json.JSONDecodeError, IOError) as e:
 logger.warning(f"Önbellek yüklenemedi: {e}")
 return {}
 return {}

 def _save_cache(self) -> None:
 """Önbelleği diske kaydet."""
 try:
 with open(self.cache_file, "w", encoding="utf-8") as f:
 json.dump(self.cache, f, ensure_ascii=False, indent=2)
 logger.debug(f"Önbellek kaydedildi: {len(self.cache)} giriş")
 except IOError as e:
 logger.error(f"Önbellek kaydedilemedi: {e}")

 @staticmethod
 def _hash_query(query: str) -> str:
 """Sorgu için benzersiz hash oluştur.

 Args:
 query: Sorgu metni.

 Returns:
 str: SHA-256 hash değeri.
 """
 normalized = query.strip().lower()
 return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

 def get(self, query: str) -> Optional[Dict[str, Any]]:
 """Önbellekten sorgu sonucunu getir.

 Args:
 query: Sorgu metni.

 Returns:
 Optional[Dict[str, Any]]: Önbellekteki cevap veya None.
 """
 query_hash = self._hash_query(query)

 if query_hash not in self.cache:
 return None

 entry = self.cache[query_hash]

 # TTL kontrolü
 cached_time = datetime.fromisoformat(entry["timestamp"])
 if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
 # Süresi dolmuş, sil
 del self.cache[query_hash]
 self._save_cache()
 logger.debug(f"Önbellek süresi dolmuş: '{query[:30]}...'")
 return None

 logger.info(f"Önbellekten döndürüldü: '{query[:30]}...'")
 print(f" {Fore.YELLOW} Önbellekten döndürüldü{Style.RESET_ALL}")

 return entry["data"]

 def set(self, query: str, data: Dict[str, Any]) -> None:
 """Sorgu sonucunu önbelleğe kaydet.

 Args:
 query: Sorgu metni.
 data: Kaydedilecek cevap verisi.
 """
 query_hash = self._hash_query(query)

 # Source documents'ı serialize edilebilir hale getir
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
 logger.debug(f"Önbelleğe kaydedildi: '{query[:30]}...'")

 def clear(self) -> int:
 """Tüm önbelleği temizle.

 Returns:
 int: Silinen giriş sayısı.
 """
 count = len(self.cache)
 self.cache = {}
 self._save_cache()
 print(f"{Fore.GREEN} ️ Önbellek temizlendi: {count} giriş silindi{Style.RESET_ALL}")
 logger.info(f"Önbellek temizlendi: {count} giriş")
 return count

 def stats(self) -> Dict[str, Any]:
 """Önbellek istatistiklerini döndür.

 Returns:
 Dict[str, Any]: İstatistik bilgileri.
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

 print(f"\n{Fore.CYAN} Önbellek İstatistikleri{Style.RESET_ALL}")
 print(f" Toplam giriş: {stats['total_entries']}")
 print(f" Geçerli: {Fore.GREEN}{stats['valid_entries']}{Style.RESET_ALL}")
 print(f" Süresi dolmuş: {Fore.YELLOW}{stats['expired_entries']}{Style.RESET_ALL}")
 print(f" TTL: {stats['ttl_hours']} saat")
 print()

 return stats
