from __future__ import annotations

import os
import sys

from streamlit.web import cli as stcli

from sap_tender_bot.config import get_repo_root


def main() -> int:
    repo_root = get_repo_root()
    app_path = repo_root / "streamlit_app.py"
    if not app_path.exists():
        raise SystemExit("streamlit_app.py not found. Run from the repo root.")

    os.chdir(repo_root)
    sys.argv = ["streamlit", "run", str(app_path)]
    try:
        stcli.main()
    except SystemExit as exc:
        return int(exc.code) if exc.code else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
