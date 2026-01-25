from __future__ import annotations

import json
import os
import sys
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import DATA_DIR, LOG_DIR  # noqa: E402

EXPORT_DIR = DATA_DIR / "exports"
STATE_PATH = DATA_DIR / "state.json"
UI_CONFIG_PATH = DATA_DIR / "ui_config.json"
UI_STATE_PATH = DATA_DIR / "ui_state.json"

DATE_FIELD_MAP = {
    "Updated": "updated_at",
    "Published": "publish_date",
    "Close": "close_date",
}


def set_page_config(title: str) -> None:
    if not st.session_state.get("_page_config_set"):
        st.set_page_config(page_title=title, layout="wide")
        st.session_state["_page_config_set"] = True


def get_env_label() -> str:
    value = os.getenv("APP_ENV")
    return value.strip() if value else "dev"


def render_sidebar() -> dict:
    st.sidebar.title("SAP Tender Bot")
    role = st.sidebar.radio("Role", ["Ops", "Admin"], key="role", horizontal=True)

    st.sidebar.subheader("Filters")
    search = st.sidebar.text_input("Search", key="search")
    use_date = st.sidebar.checkbox("Use date filter", value=False, key="use_date")
    date_field_label = st.sidebar.selectbox("Date field", list(DATE_FIELD_MAP), key="date_field")

    today = date.today()
    default_start = today.replace(day=1)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, today),
        key="date_range",
        disabled=not use_date,
    )

    score_range = st.sidebar.slider("Score range", 0, 100, (0, 100), key="score_range")

    return {
        "role": role,
        "search": search,
        "use_date": use_date,
        "date_field": DATE_FIELD_MAP.get(date_field_label, "updated_at"),
        "date_range": date_range,
        "score_range": score_range,
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    content = STATE_PATH.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _merge_dict(defaults: dict, override: dict) -> dict:
    merged = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_ui_config() -> dict:
    defaults = {
        "sources": [
            {
                "name": "CanadaBuys CKAN",
                "enabled": True,
                "priority": 1,
                "notes": "Open tender notices",
            }
        ],
        "notifications": {
            "email_recipients": [],
            "slack_webhook": "",
            "digest_schedule": "daily 08:35",
        },
        "rules": {
            "min_score_non_sap": 70,
            "max_non_sap_close_days": 365,
            "include_staffing": False,
            "include_supply_arrangements": False,
        },
    }
    if not UI_CONFIG_PATH.exists():
        return defaults
    content = UI_CONFIG_PATH.read_text(encoding="utf-8").strip()
    if not content:
        return defaults
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return defaults
    return _merge_dict(defaults, data)


def save_ui_config(config: dict) -> None:
    UI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_ui_state() -> dict:
    defaults = {
        "reviewed_uids": [],
        "review_status": {},
        "notes": {},
    }
    if not UI_STATE_PATH.exists():
        return defaults
    content = UI_STATE_PATH.read_text(encoding="utf-8").strip()
    if not content:
        return defaults
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return defaults
    return _merge_dict(defaults, data)


def save_ui_state(state: dict) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def mark_reviewed(uid: str, status: Optional[str] = None) -> None:
    if not uid:
        return
    state = load_ui_state()
    reviewed = set(state.get("reviewed_uids") or [])
    reviewed.add(uid)
    state["reviewed_uids"] = sorted(reviewed)
    if status:
        review_status = state.get("review_status", {})
        review_status[str(uid)] = status
        state["review_status"] = review_status
    save_ui_state(state)


def save_note(uid: str, note: str) -> None:
    if not uid:
        return
    state = load_ui_state()
    notes = state.get("notes", {})
    if note:
        notes[str(uid)] = note
    else:
        notes.pop(str(uid), None)
    state["notes"] = notes
    save_ui_state(state)


def latest_export(pattern: str) -> Optional[Path]:
    if not EXPORT_DIR.exists():
        return None
    files = sorted(EXPORT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def remove_uid_from_latest_csv(pattern: str, uid: str) -> bool:
    if not uid:
        return False
    path = latest_export(pattern)
    if not path or not path.exists():
        return False
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return False
    if "uid" not in df.columns:
        return False
    original_len = len(df)
    df = df[df["uid"] != uid]
    if len(df) == original_len:
        return False
    df.to_csv(path, index=False, encoding="utf-8")
    return True


def append_row_to_latest_csv(pattern: str, row: dict) -> bool:
    uid = str(row.get("uid", "")).strip()
    if not uid:
        return False

    path = latest_export(pattern)
    if not path:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPORT_DIR / f"flagged_{datetime.now():%Y%m%d}.csv"

    if path.exists():
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if not df.empty and "uid" in df.columns and uid in set(df["uid"]):
        return False

    if df.empty:
        columns = list(row.keys())
    else:
        columns = list(df.columns)

    cleaned = {}
    for col in columns:
        value = row.get(col, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        cleaned[col] = "" if value is None else value

    new_df = pd.DataFrame([cleaned], columns=columns)
    if df.empty:
        combined = new_df
    else:
        combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    return True


def load_export_df(pattern: str) -> pd.DataFrame:
    path = latest_export(pattern)
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()
    for col in ("publish_date", "amendment_date", "updated_at", "close_date"):
        if col in df.columns:
            df[f"{col}_dt"] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if "score" in df.columns:
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    df.attrs["source_path"] = str(path)
    return df


def load_reject_report(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    search = (filters.get("search") or "").strip()
    if search:
        search_lower = search.lower()
        text_cols = [c for c in ("title", "org", "id", "uid") if c in result.columns]
        if text_cols:
            mask = pd.Series(False, index=result.index)
            for col in text_cols:
                mask |= result[col].astype(str).str.lower().str.contains(search_lower, na=False)
            result = result[mask]

    score_range = filters.get("score_range") or (0, 100)
    if "score_num" in result.columns:
        min_score, max_score = score_range
        result = result[(result["score_num"] >= min_score) & (result["score_num"] <= max_score)]

    if filters.get("use_date"):
        field = filters.get("date_field") or "updated_at"
        dt_col = f"{field}_dt"
        if dt_col in result.columns:
            date_range = filters.get("date_range")
            start_date, end_date = _normalize_date_range(date_range)
            if start_date and end_date:
                start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
                end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
                result = result[(result[dt_col] >= start_dt) & (result[dt_col] <= end_dt)]

    return result


def _normalize_date_range(value: Any) -> tuple[Optional[date], Optional[date]]:
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        return start, end
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, date):
        return value, value
    return None, None


def counts_by_date(df: pd.DataFrame, field: str) -> pd.DataFrame:
    dt_col = f"{field}_dt"
    if dt_col not in df.columns:
        return pd.DataFrame()
    usable = df[df[dt_col].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    usable["date"] = usable[dt_col].dt.date
    counts = usable.groupby("date", as_index=False).size()
    counts.rename(columns={"size": "count"}, inplace=True)
    return counts


def list_log_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)


def read_log_tail(path: Path, max_lines: int = 400) -> str:
    if not path or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            tail = deque(handle, maxlen=max_lines)
        return "".join(tail)
    except Exception:
        return ""


def count_errors_in_log(path: Optional[Path]) -> int:
    if not path or not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "| ERROR |" in line:
                    count += 1
    except Exception:
        return 0
    return count


def fmt_dt(value: Optional[str]) -> str:
    if not value:
        return "n/a"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def env_status() -> dict:
    return {
        "SMTP_HOST": bool(os.getenv("SMTP_HOST")),
        "SMTP_PORT": bool(os.getenv("SMTP_PORT")),
        "SMTP_USER": bool(os.getenv("SMTP_USER")),
        "SMTP_PASS": bool(os.getenv("SMTP_PASS")),
        "EMAIL_TO": bool(os.getenv("EMAIL_TO")),
        "EMAIL_FROM": bool(os.getenv("EMAIL_FROM")),
        "COHERE_API_KEY": bool(os.getenv("COHERE_API_KEY")),
    }
