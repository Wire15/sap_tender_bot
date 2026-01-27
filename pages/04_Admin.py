from __future__ import annotations

import subprocess
import sys

import pandas as pd
import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()

st.title("Admin")

if filters.get("role") != "Admin":
    st.warning("Admin access required. Switch role in the sidebar.")
    st.stop()

config = ui.load_ui_config()
rules = config.rules

st.subheader("Run controls")
with st.form("run_controls"):
    st.caption("Run uses saved rule defaults unless overridden.")
    col1, col2, col3 = st.columns(3)
    with col1:
        dry_run = st.checkbox("Dry run (no email)", value=True)
        include_llm = st.checkbox("Include LLM enrichment", value=False)
        include_attachments = st.checkbox("Include attachment enrichment", value=True)
    with col2:
        export_csv = st.checkbox("Export CSV", value=True)
        export_near_miss = st.checkbox("Export near-miss", value=True)
        export_report = st.checkbox("Export reject report", value=True)
    with col3:
        top_n = st.number_input("Top N", min_value=1, max_value=100, value=25, step=1)
        since_iso = st.text_input("Since ISO (optional)")

    confirm_send = st.checkbox("I want to send email now", value=False)
    submitted = st.form_submit_button("Run now")

if submitted:
    if not dry_run and not confirm_send:
        st.error("Confirm email sending before running a full digest.")
    else:
        args = [sys.executable, "-m", "sap_tender_bot.daily_digest", "--top-n", str(top_n)]
        if dry_run:
            args.append("--dry-run")
        if export_csv:
            args.append("--export-csv")
        if export_near_miss:
            args.append("--export-near-miss")
        if export_report:
            args.append("--export-report")
        args.extend(
            [
                "--min-score-non-sap",
                str(int(rules.min_score_non_sap)),
                "--max-non-sap-close-days",
                str(int(rules.max_non_sap_close_days)),
            ]
        )
        if bool(rules.include_staffing):
            args.append("--include-staffing")
        else:
            args.append("--no-include-staffing")
        if bool(rules.include_supply_arrangements):
            args.append("--include-supply-arrangements")
        else:
            args.append("--no-include-supply-arrangements")
        if not include_llm:
            args.append("--no-llm")
        if not include_attachments:
            args.append("--no-attachments")
        if since_iso:
            args.extend(["--since-iso", since_iso])

        with st.spinner("Running digest..."):
            result = subprocess.run(
                args,
                cwd=str(ui.REPO_ROOT),
                capture_output=True,
                text=True,
            )
        st.code(result.stdout or "(no stdout)")
        if result.stderr:
            st.error(result.stderr)
        st.info(f"Exit code: {result.returncode}")

st.subheader("Reprocess by ID")
st.write("Single-tender reprocessing is not implemented yet.")

st.subheader("Source management")
sources_df = pd.DataFrame(config.sources)
edited_sources = st.data_editor(sources_df, num_rows="dynamic", use_container_width=True)
if st.button("Save sources"):
    config.sources = edited_sources.to_dict(orient="records")
    ui.save_ui_config(config)
    st.success("Sources saved")

st.subheader("Rule defaults")
min_score = st.number_input(
    "Min score (non-SAP)",
    min_value=0,
    max_value=100,
    value=int(rules.min_score_non_sap),
    step=1,
)
max_close_days = st.number_input(
    "Max close days (non-SAP)",
    min_value=1,
    max_value=3650,
    value=int(rules.max_non_sap_close_days),
    step=1,
)
include_staffing = st.checkbox(
    "Include staffing",
    value=bool(rules.include_staffing),
)
include_supply = st.checkbox(
    "Include supply arrangements",
    value=bool(rules.include_supply_arrangements),
)

if st.button("Save rule defaults"):
    config.rules.min_score_non_sap = int(min_score)
    config.rules.max_non_sap_close_days = int(max_close_days)
    config.rules.include_staffing = bool(include_staffing)
    config.rules.include_supply_arrangements = bool(include_supply)
    ui.save_ui_config(config)
    st.success("Rule defaults saved")
