from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class CKANClient:
    base_url: str = "https://open.canada.ca/data/en/api/3/action"
    timeout_s: int = 60

    def _get(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{action}"
        headers = {
            "User-Agent": "sap-tender-bot/0.1 (+https://github.com/yourname/sap-tender-bot)"
        }

        # simple retry loop (handles transient 5xx, timeouts)
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                if not data.get("success", False):
                    raise RuntimeError(f"CKAN action failed: {data}")
                return data["result"]
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"CKAN request failed after retries: {last_err}")

    def package_show(self, dataset_id: str) -> Dict[str, Any]:
        return self._get("package_show", {"id": dataset_id})
