from __future__ import annotations

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
ui.render_sidebar()

st.title("Logs")

log_files = ui.list_log_files()
if not log_files:
    st.info("No log files found yet.")
    st.stop()

selected = st.selectbox("Select log file", log_files, format_func=lambda p: p.name)

level_filter = st.multiselect(
    "Filter levels",
    ["INFO", "WARNING", "ERROR"],
    default=["INFO", "WARNING", "ERROR"],
)

tail = ui.read_log_tail(selected, max_lines=400)
if level_filter:
    filtered = []
    for line in tail.splitlines():
        if any(f"| {level} |" in line for level in level_filter):
            filtered.append(line)
    tail = "\n".join(filtered)

st.text(tail if tail else "No matching log lines.")

with selected.open("rb") as handle:
    st.download_button("Download log", data=handle.read(), file_name=selected.name)
