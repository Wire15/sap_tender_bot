from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

import ui_common as ui


def main() -> None:
    ui.set_page_config("SAP Tender Admin")
    ui.render_sidebar()
    state = ui.load_state()

    st.title("Dashboard")
    env_label = ui.get_env_label()
    last_run = ui.fmt_dt(state.get("last_run_iso"))

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.caption(f"Environment: {env_label}")
    with top_right:
        st.caption(f"Last run: {last_run}")

    flagged_df = ui.load_export_df("flagged_*.csv")
    near_df = ui.load_export_df("near_miss_*.csv")
    report_path = ui.latest_export("reject_report_*.json")
    report = ui.load_reject_report(report_path)

    log_files = ui.list_log_files()
    latest_log = log_files[0] if log_files else None
    error_count = ui.count_errors_in_log(latest_log)

    totals = report.get("totals", {})
    total_new = totals.get("new")
    total_flagged = totals.get("flagged", len(flagged_df))
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Flagged", total_flagged)
    kpi2.metric("Review queue", len(near_df))
    kpi3.metric("Errors (latest log)", error_count)
    kpi4.metric("New items", total_new if total_new is not None else "n/a")

    st.subheader("Recent exports")
    exports = [
        ("Flagged CSV", ui.latest_export("flagged_*.csv")),
        ("Near-miss CSV", ui.latest_export("near_miss_*.csv")),
        ("Reject report", report_path),
    ]
    rows = []
    for label, path in exports:
        if path and path.exists():
            rows.append(
                {
                    "Export": label,
                    "File": path.name,
                    "Modified": path.stat().st_mtime,
                }
            )
        else:
            rows.append({"Export": label, "File": "n/a", "Modified": "n/a"})
    if rows:
        display = []
        for row in rows:
            mtime = row["Modified"]
            if isinstance(mtime, float):
                mtime = ui.fmt_dt(
                    datetime.utcfromtimestamp(row["Modified"]).replace(tzinfo=timezone.utc).isoformat()
                )
            display.append({"Export": row["Export"], "File": row["File"], "Modified": mtime})
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Flagged trend")
    chart = ui.counts_by_date(flagged_df, "updated_at")
    if chart.empty:
        st.info("No flagged trend data yet.")
    else:
        st.line_chart(chart.set_index("date"))

    st.subheader("Latest log tail")
    if latest_log:
        with st.expander(f"{latest_log.name}", expanded=False):
            st.text(ui.read_log_tail(latest_log, max_lines=200))
    else:
        st.info("No log files found.")


if __name__ == "__main__":
    main()
