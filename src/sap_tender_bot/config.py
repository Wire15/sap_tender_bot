from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

DEFAULT_WATCHLIST_BLOCKLIST = [
    "exclude_supply_arrangement",
    "exclude_staffing",
    "exclude_negative_title",
    "exclude_non_it_unspsc",
    "exclude_goods_or_hardware",
    "exclude_no_erp_domain",
    "exclude_far_future_close",
]

DEFAULT_SOURCES = [
    {
        "name": "CanadaBuys CKAN",
        "enabled": True,
        "priority": 1,
        "notes": "Open tender notices",
    }
]


def _bool_from_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_from_env(value: Optional[str], default: int) -> int:
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _list_from_env(value: Optional[str], default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _ensure_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(default)


def _resolve_path(repo_root: Path, value: Optional[str], default: str) -> Path:
    raw = value if value else default
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_repo_root(start: Optional[Path] = None) -> Path:
    env_root = os.getenv("SAP_TENDER_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates = []
    if start:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())

    for base in candidates:
        for parent in [base] + list(base.parents):
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                return parent

    return Path.cwd().resolve()


def get_config_path(repo_root: Optional[Path] = None) -> Path:
    env_path = os.getenv("SAP_TENDER_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    repo_root = repo_root or get_repo_root()
    return repo_root / "config.yaml"


@dataclass
class PathsConfig:
    repo_root: Path
    data_dir: Path
    cache_dir: Path
    log_dir: Path
    db_path: Path

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def ui_state_path(self) -> Path:
        return self.data_dir / "ui_state.json"

    def to_dict(self) -> dict:
        return {
            "data_dir": _relativize_path(self.data_dir, self.repo_root),
            "cache_dir": _relativize_path(self.cache_dir, self.repo_root),
            "log_dir": _relativize_path(self.log_dir, self.repo_root),
            "db_path": _relativize_path(self.db_path, self.repo_root),
        }


@dataclass
class RulesConfig:
    min_score_non_sap: int = 70
    max_non_sap_close_days: int = 365
    include_staffing: bool = False
    include_supply_arrangements: bool = False
    store_hits: bool = True

    def to_dict(self) -> dict:
        return {
            "min_score_non_sap": self.min_score_non_sap,
            "max_non_sap_close_days": self.max_non_sap_close_days,
            "include_staffing": self.include_staffing,
            "include_supply_arrangements": self.include_supply_arrangements,
            "store_hits": self.store_hits,
        }


@dataclass
class DigestConfig:
    seen_retention_days: int = 180
    watchlist_max: int = 15
    watchlist_blocklist: list[str] = field(
        default_factory=lambda: list(DEFAULT_WATCHLIST_BLOCKLIST)
    )

    def to_dict(self) -> dict:
        return {
            "seen_retention_days": self.seen_retention_days,
            "watchlist_max": self.watchlist_max,
            "watchlist_blocklist": list(self.watchlist_blocklist),
        }


@dataclass
class NotificationsConfig:
    email_recipients: list[str] = field(default_factory=list)
    email_from: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    slack_webhook: str = ""
    digest_schedule: str = "daily 08:35"

    def to_dict(self) -> dict:
        return {
            "email_recipients": list(self.email_recipients),
            "email_from": self.email_from,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_user": self.smtp_user,
            "smtp_pass": self.smtp_pass,
            "slack_webhook": self.slack_webhook,
            "digest_schedule": self.digest_schedule,
        }


@dataclass
class LLMConfig:
    enabled: bool = True
    api_key: str = ""
    model: str = "command-a-03-2025"
    api_base: str = "https://api.cohere.com"
    timeout_s: int = 45
    borderline_max: int = 10
    daily_cap: int = 120
    per_run_max: int = 40
    watchlist_min_score: int = 80

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "api_key": self.api_key,
            "model": self.model,
            "api_base": self.api_base,
            "timeout_s": self.timeout_s,
            "borderline_max": self.borderline_max,
            "daily_cap": self.daily_cap,
            "per_run_max": self.per_run_max,
            "watchlist_min_score": self.watchlist_min_score,
        }


@dataclass
class EmbeddingsConfig:
    enabled: bool = True
    api_key: str = ""
    model: str = "embed-multilingual-v3.0"
    api_base: str = "https://api.cohere.com"
    timeout_s: int = 45
    batch_size: int = 32
    input_type: str = "search_document"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "api_key": self.api_key,
            "model": self.model,
            "api_base": self.api_base,
            "timeout_s": self.timeout_s,
            "batch_size": self.batch_size,
            "input_type": self.input_type,
        }


@dataclass
class AttachmentsConfig:
    enabled: bool = True
    max_attachments_per_run: int = 20
    max_attachments_per_tender: int = 3
    max_bytes: int = 15_000_000
    timeout_s: int = 40
    download_retries: int = 2
    download_concurrency: int = 4
    section_char_budget: int = 4000
    max_sections: int = 4
    min_text_chars_for_ocr: int = 800
    ocr_enabled: bool = False
    pii_scrub: bool = False
    spreadsheets_enabled: bool = True
    spreadsheet_max_bytes: int = 8_000_000
    spreadsheet_max_rows: int = 2000
    spreadsheet_max_cols: int = 40
    spreadsheet_max_sheets: int = 5
    spreadsheet_sheet_char_budget: int = 2000
    docx_tables_enabled: bool = True
    docx_table_max_tables: int = 6
    docx_table_max_rows: int = 200
    docx_table_max_cols: int = 20
    docx_table_char_budget: int = 3000
    chunked_retrieval_enabled: bool = False
    chunk_text_max_chars: int = 1200
    chunk_text_overlap: int = 100
    chunk_char_budget: int = 3200
    chunk_max_per_attachment: int = 6
    chunk_max_per_tender: int = 18
    chunk_max_per_run: int = 60
    structured_summaries_enabled: bool = False
    language_detection_enabled: bool = False
    language_default: str = "en"
    llm_enabled: bool = True
    llm_kill_switch: bool = False
    llm_model: str = "command-a-03-2025"
    llm_api_key: str = ""
    llm_api_base: str = "https://api.cohere.com"
    llm_timeout_s: int = 45
    llm_per_run_max: int = 20
    llm_concurrency: int = 2
    llm_daily_cap: int = 0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_attachments_per_run": self.max_attachments_per_run,
            "max_attachments_per_tender": self.max_attachments_per_tender,
            "max_bytes": self.max_bytes,
            "timeout_s": self.timeout_s,
            "download_retries": self.download_retries,
            "download_concurrency": self.download_concurrency,
            "section_char_budget": self.section_char_budget,
            "max_sections": self.max_sections,
            "min_text_chars_for_ocr": self.min_text_chars_for_ocr,
            "ocr_enabled": self.ocr_enabled,
            "pii_scrub": self.pii_scrub,
            "spreadsheets_enabled": self.spreadsheets_enabled,
            "spreadsheet_max_bytes": self.spreadsheet_max_bytes,
            "spreadsheet_max_rows": self.spreadsheet_max_rows,
            "spreadsheet_max_cols": self.spreadsheet_max_cols,
            "spreadsheet_max_sheets": self.spreadsheet_max_sheets,
            "spreadsheet_sheet_char_budget": self.spreadsheet_sheet_char_budget,
            "docx_tables_enabled": self.docx_tables_enabled,
            "docx_table_max_tables": self.docx_table_max_tables,
            "docx_table_max_rows": self.docx_table_max_rows,
            "docx_table_max_cols": self.docx_table_max_cols,
            "docx_table_char_budget": self.docx_table_char_budget,
            "chunked_retrieval_enabled": self.chunked_retrieval_enabled,
            "chunk_text_max_chars": self.chunk_text_max_chars,
            "chunk_text_overlap": self.chunk_text_overlap,
            "chunk_char_budget": self.chunk_char_budget,
            "chunk_max_per_attachment": self.chunk_max_per_attachment,
            "chunk_max_per_tender": self.chunk_max_per_tender,
            "chunk_max_per_run": self.chunk_max_per_run,
            "structured_summaries_enabled": self.structured_summaries_enabled,
            "language_detection_enabled": self.language_detection_enabled,
            "language_default": self.language_default,
            "llm_enabled": self.llm_enabled,
            "llm_kill_switch": self.llm_kill_switch,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "llm_api_base": self.llm_api_base,
            "llm_timeout_s": self.llm_timeout_s,
            "llm_per_run_max": self.llm_per_run_max,
            "llm_concurrency": self.llm_concurrency,
            "llm_daily_cap": self.llm_daily_cap,
        }


@dataclass
class AppConfig:
    app_env: str = "dev"
    timezone: str = "America/Toronto"
    paths: PathsConfig = field(
        default_factory=lambda: build_paths(get_repo_root(), {}, apply_defaults=True)
    )
    rules: RulesConfig = field(default_factory=RulesConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    attachments: AttachmentsConfig = field(default_factory=AttachmentsConfig)
    sources: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_SOURCES))

    def to_dict(self) -> dict:
        return {
            "app": {"env": self.app_env, "timezone": self.timezone},
            "paths": self.paths.to_dict(),
            "rules": self.rules.to_dict(),
            "digest": self.digest.to_dict(),
            "notifications": self.notifications.to_dict(),
            "llm": self.llm.to_dict(),
            "embeddings": self.embeddings.to_dict(),
            "attachments": self.attachments.to_dict(),
            "sources": list(self.sources),
        }


def _relativize_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def build_paths(repo_root: Path, data: dict, apply_defaults: bool = True) -> PathsConfig:
    data_dir = _resolve_path(repo_root, data.get("data_dir") if data else None, "data")
    cache_default = str(data_dir / "cache") if apply_defaults else "data/cache"
    cache_dir = _resolve_path(repo_root, data.get("cache_dir") if data else None, cache_default)
    log_dir = _resolve_path(repo_root, data.get("log_dir") if data else None, "logs")
    db_default = str(data_dir / "sap_tender.db") if apply_defaults else "data/sap_tender.db"
    db_path = _resolve_path(repo_root, data.get("db_path") if data else None, db_default)
    return PathsConfig(
        repo_root=repo_root,
        data_dir=data_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        db_path=db_path,
    )


def default_config_dict(repo_root: Path) -> dict:
    return {
        "app": {"env": "dev", "timezone": "America/Toronto"},
        "paths": {
            "data_dir": "data",
            "cache_dir": "data/cache",
            "log_dir": "logs",
            "db_path": "data/sap_tender.db",
        },
        "rules": RulesConfig().to_dict(),
        "digest": DigestConfig().to_dict(),
        "notifications": NotificationsConfig().to_dict(),
        "llm": LLMConfig().to_dict(),
        "embeddings": EmbeddingsConfig().to_dict(),
        "attachments": AttachmentsConfig().to_dict(),
        "sources": list(DEFAULT_SOURCES),
    }


def config_from_dict(repo_root: Path, data: dict) -> AppConfig:
    app = data.get("app", {}) if isinstance(data, dict) else {}
    paths_data = data.get("paths", {}) if isinstance(data, dict) else {}
    rules_data = data.get("rules", {}) if isinstance(data, dict) else {}
    digest_data = data.get("digest", {}) if isinstance(data, dict) else {}
    notifications_data = data.get("notifications", {}) if isinstance(data, dict) else {}
    llm_data = data.get("llm", {}) if isinstance(data, dict) else {}
    embeddings_data = data.get("embeddings", {}) if isinstance(data, dict) else {}
    attachments_data = data.get("attachments", {}) if isinstance(data, dict) else {}
    sources_data = data.get("sources", None)

    config = AppConfig(
        app_env=str(app.get("env", "dev")),
        timezone=str(app.get("timezone", "America/Toronto")),
        paths=build_paths(repo_root, paths_data, apply_defaults=True),
        rules=RulesConfig(
            min_score_non_sap=int(rules_data.get("min_score_non_sap", 70)),
            max_non_sap_close_days=int(rules_data.get("max_non_sap_close_days", 365)),
            include_staffing=bool(rules_data.get("include_staffing", False)),
            include_supply_arrangements=bool(rules_data.get("include_supply_arrangements", False)),
            store_hits=bool(rules_data.get("store_hits", True)),
        ),
        digest=DigestConfig(
            seen_retention_days=int(digest_data.get("seen_retention_days", 180)),
            watchlist_max=int(digest_data.get("watchlist_max", 15)),
            watchlist_blocklist=_ensure_list(
                digest_data.get("watchlist_blocklist", list(DEFAULT_WATCHLIST_BLOCKLIST)),
                list(DEFAULT_WATCHLIST_BLOCKLIST),
            ),
        ),
        notifications=NotificationsConfig(
            email_recipients=_ensure_list(notifications_data.get("email_recipients", []), []),
            email_from=str(notifications_data.get("email_from", "")),
            smtp_host=str(notifications_data.get("smtp_host", "")),
            smtp_port=int(notifications_data.get("smtp_port", 587)),
            smtp_user=str(notifications_data.get("smtp_user", "")),
            smtp_pass=str(notifications_data.get("smtp_pass", "")),
            slack_webhook=str(notifications_data.get("slack_webhook", "")),
            digest_schedule=str(notifications_data.get("digest_schedule", "daily 08:35")),
        ),
        llm=LLMConfig(
            enabled=bool(llm_data.get("enabled", True)),
            api_key=str(llm_data.get("api_key", "")),
            model=str(llm_data.get("model", "command-r")),
            api_base=str(llm_data.get("api_base", "https://api.cohere.ai")),
            timeout_s=int(llm_data.get("timeout_s", 45)),
            borderline_max=int(llm_data.get("borderline_max", 10)),
            daily_cap=int(llm_data.get("daily_cap", 120)),
            per_run_max=int(llm_data.get("per_run_max", 40)),
            watchlist_min_score=int(llm_data.get("watchlist_min_score", 80)),
        ),
        embeddings=EmbeddingsConfig(
            enabled=bool(embeddings_data.get("enabled", True)),
            api_key=str(embeddings_data.get("api_key", "")),
            model=str(embeddings_data.get("model", "embed-multilingual-v3.0")),
            api_base=str(embeddings_data.get("api_base", "https://api.cohere.ai")),
            timeout_s=int(embeddings_data.get("timeout_s", 45)),
            batch_size=int(embeddings_data.get("batch_size", 32)),
            input_type=str(embeddings_data.get("input_type", "search_document")),
        ),
        attachments=AttachmentsConfig(
            enabled=bool(attachments_data.get("enabled", True)),
            max_attachments_per_run=int(attachments_data.get("max_attachments_per_run", 20)),
            max_attachments_per_tender=int(attachments_data.get("max_attachments_per_tender", 3)),
            max_bytes=int(attachments_data.get("max_bytes", 15_000_000)),
            timeout_s=int(attachments_data.get("timeout_s", 40)),
            download_retries=int(attachments_data.get("download_retries", 2)),
            download_concurrency=int(attachments_data.get("download_concurrency", 4)),
            section_char_budget=int(attachments_data.get("section_char_budget", 4000)),
            max_sections=int(attachments_data.get("max_sections", 4)),
            min_text_chars_for_ocr=int(attachments_data.get("min_text_chars_for_ocr", 800)),
            ocr_enabled=bool(attachments_data.get("ocr_enabled", False)),
            pii_scrub=bool(attachments_data.get("pii_scrub", False)),
            spreadsheets_enabled=bool(attachments_data.get("spreadsheets_enabled", True)),
            spreadsheet_max_bytes=int(attachments_data.get("spreadsheet_max_bytes", 8_000_000)),
            spreadsheet_max_rows=int(attachments_data.get("spreadsheet_max_rows", 2000)),
            spreadsheet_max_cols=int(attachments_data.get("spreadsheet_max_cols", 40)),
            spreadsheet_max_sheets=int(attachments_data.get("spreadsheet_max_sheets", 5)),
            spreadsheet_sheet_char_budget=int(
                attachments_data.get("spreadsheet_sheet_char_budget", 2000)
            ),
            docx_tables_enabled=bool(attachments_data.get("docx_tables_enabled", True)),
            docx_table_max_tables=int(attachments_data.get("docx_table_max_tables", 6)),
            docx_table_max_rows=int(attachments_data.get("docx_table_max_rows", 200)),
            docx_table_max_cols=int(attachments_data.get("docx_table_max_cols", 20)),
            docx_table_char_budget=int(attachments_data.get("docx_table_char_budget", 3000)),
            chunked_retrieval_enabled=bool(
                attachments_data.get("chunked_retrieval_enabled", False)
            ),
            chunk_text_max_chars=int(attachments_data.get("chunk_text_max_chars", 1200)),
            chunk_text_overlap=int(attachments_data.get("chunk_text_overlap", 100)),
            chunk_char_budget=int(attachments_data.get("chunk_char_budget", 3200)),
            chunk_max_per_attachment=int(
                attachments_data.get("chunk_max_per_attachment", 6)
            ),
            chunk_max_per_tender=int(attachments_data.get("chunk_max_per_tender", 18)),
            chunk_max_per_run=int(attachments_data.get("chunk_max_per_run", 60)),
            structured_summaries_enabled=bool(
                attachments_data.get("structured_summaries_enabled", False)
            ),
            language_detection_enabled=bool(
                attachments_data.get("language_detection_enabled", False)
            ),
            language_default=str(attachments_data.get("language_default", "en")),
            llm_enabled=bool(attachments_data.get("llm_enabled", True)),
            llm_kill_switch=bool(attachments_data.get("llm_kill_switch", False)),
            llm_model=str(attachments_data.get("llm_model", "command-r")),
            llm_api_key=str(attachments_data.get("llm_api_key", "")),
            llm_api_base=str(attachments_data.get("llm_api_base", "https://api.cohere.ai")),
            llm_timeout_s=int(attachments_data.get("llm_timeout_s", 45)),
            llm_per_run_max=int(attachments_data.get("llm_per_run_max", 20)),
            llm_concurrency=int(attachments_data.get("llm_concurrency", 2)),
            llm_daily_cap=int(attachments_data.get("llm_daily_cap", 0)),
        ),
        sources=list(sources_data) if isinstance(sources_data, list) else list(DEFAULT_SOURCES),
    )

    return config


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    data = yaml.safe_load(content)
    return data if isinstance(data, dict) else {}


def load_config(
    path: Optional[Path] = None,
    apply_env: bool = True,
    load_env_file: bool = True,
) -> AppConfig:
    repo_root = get_repo_root()
    config_path = path or get_config_path(repo_root)

    if load_env_file:
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    defaults = default_config_dict(repo_root)
    yaml_data = _load_yaml(config_path)
    merged = _deep_merge(defaults, yaml_data)
    config = config_from_dict(repo_root, merged)

    if apply_env:
        apply_env_overrides(config)

    return config


def save_config(config: AppConfig, path: Optional[Path] = None) -> Path:
    repo_root = config.paths.repo_root
    config_path = path or get_config_path(repo_root)
    payload = config.to_dict()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return config_path


def apply_env_overrides(config: AppConfig) -> AppConfig:
    config.app_env = os.getenv("APP_ENV", config.app_env)
    config.timezone = os.getenv("SAP_TENDER_TIMEZONE", config.timezone)

    data_dir = os.getenv("SAP_TENDER_DATA_DIR")
    cache_dir = os.getenv("SAP_TENDER_CACHE_DIR")
    log_dir = os.getenv("SAP_TENDER_LOG_DIR")
    db_path = os.getenv("SAP_TENDER_DB_PATH")
    if data_dir or cache_dir or log_dir or db_path:
        repo_root = config.paths.repo_root
        paths_data = {
            "data_dir": data_dir or _relativize_path(config.paths.data_dir, repo_root),
            "cache_dir": cache_dir or _relativize_path(config.paths.cache_dir, repo_root),
            "log_dir": log_dir or _relativize_path(config.paths.log_dir, repo_root),
            "db_path": db_path or _relativize_path(config.paths.db_path, repo_root),
        }
        config.paths = build_paths(repo_root, paths_data, apply_defaults=True)

    config.rules.min_score_non_sap = _int_from_env(
        os.getenv("SAP_TENDER_MIN_SCORE_NON_SAP"),
        config.rules.min_score_non_sap,
    )
    config.rules.max_non_sap_close_days = _int_from_env(
        os.getenv("SAP_TENDER_MAX_NON_SAP_CLOSE_DAYS"),
        config.rules.max_non_sap_close_days,
    )
    config.rules.include_staffing = _bool_from_env(
        os.getenv("SAP_TENDER_INCLUDE_STAFFING"),
        config.rules.include_staffing,
    )
    config.rules.include_supply_arrangements = _bool_from_env(
        os.getenv("SAP_TENDER_INCLUDE_SUPPLY_ARRANGEMENTS"),
        config.rules.include_supply_arrangements,
    )
    config.rules.store_hits = _bool_from_env(
        os.getenv("SAP_TENDER_STORE_HITS"),
        config.rules.store_hits,
    )

    config.digest.seen_retention_days = _int_from_env(
        os.getenv("SAP_TENDER_SEEN_RETENTION_DAYS"),
        config.digest.seen_retention_days,
    )
    config.digest.watchlist_max = _int_from_env(
        os.getenv("SAP_TENDER_WATCHLIST_MAX"),
        config.digest.watchlist_max,
    )
    config.digest.watchlist_blocklist = _list_from_env(
        os.getenv("SAP_TENDER_WATCHLIST_BLOCKLIST"),
        config.digest.watchlist_blocklist,
    )

    config.notifications.smtp_host = os.getenv("SMTP_HOST", config.notifications.smtp_host)
    config.notifications.smtp_port = _int_from_env(
        os.getenv("SMTP_PORT"),
        config.notifications.smtp_port,
    )
    config.notifications.smtp_user = os.getenv("SMTP_USER", config.notifications.smtp_user)
    config.notifications.smtp_pass = os.getenv("SMTP_PASS", config.notifications.smtp_pass)
    config.notifications.email_from = os.getenv("EMAIL_FROM", config.notifications.email_from)
    config.notifications.email_recipients = _list_from_env(
        os.getenv("EMAIL_TO"),
        config.notifications.email_recipients,
    )
    config.notifications.digest_schedule = os.getenv(
        "SAP_TENDER_DIGEST_SCHEDULE",
        config.notifications.digest_schedule,
    )

    config.llm.api_key = os.getenv("COHERE_API_KEY", config.llm.api_key)
    config.llm.model = os.getenv("COHERE_MODEL", config.llm.model)
    config.llm.api_base = os.getenv("COHERE_API_BASE", config.llm.api_base)
    config.llm.timeout_s = _int_from_env(
        os.getenv("COHERE_TIMEOUT"),
        config.llm.timeout_s,
    )
    config.llm.enabled = _bool_from_env(
        os.getenv("SAP_TENDER_LLM_ENABLED"),
        config.llm.enabled,
    )
    config.llm.borderline_max = _int_from_env(
        os.getenv("SAP_TENDER_LLM_BORDERLINE_MAX"),
        config.llm.borderline_max,
    )
    config.llm.daily_cap = _int_from_env(
        os.getenv("SAP_TENDER_LLM_DAILY_CAP"),
        config.llm.daily_cap,
    )
    config.llm.per_run_max = _int_from_env(
        os.getenv("SAP_TENDER_LLM_PER_RUN_MAX"),
        config.llm.per_run_max,
    )
    config.llm.watchlist_min_score = _int_from_env(
        os.getenv("SAP_TENDER_LLM_WATCHLIST_MIN_SCORE"),
        config.llm.watchlist_min_score,
    )

    config.embeddings.api_key = os.getenv(
        "COHERE_EMBED_API_KEY",
        os.getenv("COHERE_API_KEY", config.embeddings.api_key),
    )
    config.embeddings.model = os.getenv("COHERE_EMBED_MODEL", config.embeddings.model)
    config.embeddings.api_base = os.getenv("COHERE_EMBED_API_BASE", config.embeddings.api_base)
    config.embeddings.timeout_s = _int_from_env(
        os.getenv("COHERE_EMBED_TIMEOUT"),
        config.embeddings.timeout_s,
    )
    config.embeddings.batch_size = _int_from_env(
        os.getenv("COHERE_EMBED_BATCH_SIZE"),
        config.embeddings.batch_size,
    )
    config.embeddings.input_type = os.getenv(
        "COHERE_EMBED_INPUT_TYPE",
        config.embeddings.input_type,
    )
    config.embeddings.enabled = _bool_from_env(
        os.getenv("SAP_TENDER_EMBEDDINGS_ENABLED"),
        config.embeddings.enabled,
    )

    config.attachments.enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_ENABLED"),
        config.attachments.enabled,
    )
    config.attachments.max_attachments_per_run = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_PER_RUN_MAX"),
        config.attachments.max_attachments_per_run,
    )
    config.attachments.max_attachments_per_tender = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_PER_TENDER_MAX"),
        config.attachments.max_attachments_per_tender,
    )
    config.attachments.max_bytes = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_MAX_BYTES"),
        config.attachments.max_bytes,
    )
    config.attachments.timeout_s = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_TIMEOUT_S"),
        config.attachments.timeout_s,
    )
    config.attachments.download_retries = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOWNLOAD_RETRIES"),
        config.attachments.download_retries,
    )
    config.attachments.download_concurrency = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOWNLOAD_CONCURRENCY"),
        config.attachments.download_concurrency,
    )
    config.attachments.section_char_budget = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SECTION_CHAR_BUDGET"),
        config.attachments.section_char_budget,
    )
    config.attachments.max_sections = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_MAX_SECTIONS"),
        config.attachments.max_sections,
    )
    config.attachments.min_text_chars_for_ocr = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_MIN_TEXT_CHARS_FOR_OCR"),
        config.attachments.min_text_chars_for_ocr,
    )
    config.attachments.ocr_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_OCR_ENABLED"),
        config.attachments.ocr_enabled,
    )
    config.attachments.pii_scrub = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_PII_SCRUB"),
        config.attachments.pii_scrub,
    )
    config.attachments.spreadsheets_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEETS_ENABLED"),
        config.attachments.spreadsheets_enabled,
    )
    config.attachments.spreadsheet_max_bytes = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEET_MAX_BYTES"),
        config.attachments.spreadsheet_max_bytes,
    )
    config.attachments.spreadsheet_max_rows = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEET_MAX_ROWS"),
        config.attachments.spreadsheet_max_rows,
    )
    config.attachments.spreadsheet_max_cols = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEET_MAX_COLS"),
        config.attachments.spreadsheet_max_cols,
    )
    config.attachments.spreadsheet_max_sheets = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEET_MAX_SHEETS"),
        config.attachments.spreadsheet_max_sheets,
    )
    config.attachments.spreadsheet_sheet_char_budget = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_SPREADSHEET_SHEET_CHAR_BUDGET"),
        config.attachments.spreadsheet_sheet_char_budget,
    )
    config.attachments.docx_tables_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOCX_TABLES_ENABLED"),
        config.attachments.docx_tables_enabled,
    )
    config.attachments.docx_table_max_tables = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOCX_TABLE_MAX_TABLES"),
        config.attachments.docx_table_max_tables,
    )
    config.attachments.docx_table_max_rows = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOCX_TABLE_MAX_ROWS"),
        config.attachments.docx_table_max_rows,
    )
    config.attachments.docx_table_max_cols = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOCX_TABLE_MAX_COLS"),
        config.attachments.docx_table_max_cols,
    )
    config.attachments.docx_table_char_budget = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_DOCX_TABLE_CHAR_BUDGET"),
        config.attachments.docx_table_char_budget,
    )
    config.attachments.chunked_retrieval_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNKED_RETRIEVAL_ENABLED"),
        config.attachments.chunked_retrieval_enabled,
    )
    config.attachments.chunk_text_max_chars = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_TEXT_MAX_CHARS"),
        config.attachments.chunk_text_max_chars,
    )
    config.attachments.chunk_text_overlap = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_TEXT_OVERLAP"),
        config.attachments.chunk_text_overlap,
    )
    config.attachments.chunk_char_budget = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_CHAR_BUDGET"),
        config.attachments.chunk_char_budget,
    )
    config.attachments.chunk_max_per_attachment = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_MAX_PER_ATTACHMENT"),
        config.attachments.chunk_max_per_attachment,
    )
    config.attachments.chunk_max_per_tender = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_MAX_PER_TENDER"),
        config.attachments.chunk_max_per_tender,
    )
    config.attachments.chunk_max_per_run = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_CHUNK_MAX_PER_RUN"),
        config.attachments.chunk_max_per_run,
    )
    config.attachments.structured_summaries_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_STRUCTURED_SUMMARIES_ENABLED"),
        config.attachments.structured_summaries_enabled,
    )
    config.attachments.language_detection_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LANGUAGE_DETECTION_ENABLED"),
        config.attachments.language_detection_enabled,
    )
    config.attachments.language_default = os.getenv(
        "SAP_TENDER_ATTACHMENTS_LANGUAGE_DEFAULT",
        config.attachments.language_default,
    )
    config.attachments.llm_enabled = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_ENABLED"),
        config.attachments.llm_enabled,
    )
    config.attachments.llm_kill_switch = _bool_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_KILL_SWITCH"),
        config.attachments.llm_kill_switch,
    )
    config.attachments.llm_model = os.getenv(
        "SAP_TENDER_ATTACHMENTS_LLM_MODEL",
        config.attachments.llm_model,
    )
    config.attachments.llm_api_key = os.getenv(
        "SAP_TENDER_ATTACHMENTS_LLM_API_KEY",
        config.attachments.llm_api_key,
    )
    config.attachments.llm_api_base = os.getenv(
        "SAP_TENDER_ATTACHMENTS_LLM_API_BASE",
        config.attachments.llm_api_base,
    )
    config.attachments.llm_timeout_s = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_TIMEOUT_S"),
        config.attachments.llm_timeout_s,
    )
    config.attachments.llm_per_run_max = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_PER_RUN_MAX"),
        config.attachments.llm_per_run_max,
    )
    config.attachments.llm_concurrency = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_CONCURRENCY"),
        config.attachments.llm_concurrency,
    )
    config.attachments.llm_daily_cap = _int_from_env(
        os.getenv("SAP_TENDER_ATTACHMENTS_LLM_DAILY_CAP"),
        config.attachments.llm_daily_cap,
    )

    if config.attachments.llm_kill_switch:
        config.attachments.llm_enabled = False

    if not config.attachments.llm_api_key:
        config.attachments.llm_api_key = config.llm.api_key
    if not config.attachments.llm_api_base:
        config.attachments.llm_api_base = config.llm.api_base
    if not config.attachments.llm_timeout_s:
        config.attachments.llm_timeout_s = config.llm.timeout_s

    return config
