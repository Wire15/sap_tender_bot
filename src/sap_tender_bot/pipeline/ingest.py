from __future__ import annotations

from typing import Dict, List, Optional

from sap_tender_bot.config import AppConfig, load_config
from sap_tender_bot.connectors.canadabuys_tender_notices import (
    CanadaBuysTenderNoticesConnector,
)


def ingest(
    since_iso: Optional[str] = None,
    config: AppConfig | None = None,
) -> List[Dict[str, object]]:
    settings = config or load_config()
    connector = CanadaBuysTenderNoticesConnector(cache_dir=settings.paths.cache_dir)
    return connector.fetch_open_tender_notices(since_iso=since_iso)

