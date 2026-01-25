from __future__ import annotations

from typing import Dict, List, Optional

from connectors.canadabuys_tender_notices import CanadaBuysTenderNoticesConnector


def ingest(since_iso: Optional[str] = None) -> List[Dict[str, object]]:
    connector = CanadaBuysTenderNoticesConnector()
    return connector.fetch_open_tender_notices(since_iso=since_iso)

