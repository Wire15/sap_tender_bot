from __future__ import annotations

import json

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()

st.title("Review queue")

df = ui.load_export_df("near_miss_*.csv")
df = ui.apply_filters(df, filters)

ui_state = ui.load_ui_state()
reviewed = set(ui_state.get("reviewed_uids") or [])
review_status = ui_state.get("review_status", {})

if "uid" in df.columns:
    df["reviewed"] = df["uid"].isin(reviewed)
    df["status"] = df["uid"].map(review_status).fillna("")

if df.empty:
    st.info("No near-miss items found. Run a dry-run export to populate data.")
    st.stop()

display_cols = [
    c
    for c in ["reviewed", "status", "title", "org", "_reject_reason", "updated_at", "close_date", "url"]
    if c in df.columns
]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

df = df.reset_index(drop=True)
df["label"] = df.apply(
    lambda row: " | ".join(
        [
            str(row.get("title", "")).strip() or "Untitled",
            str(row.get("org", "")).strip() or "Unknown org",
            str(row.get("uid", "")).strip(),
        ]
    ),
    axis=1,
)

selected_label = st.selectbox("Select a tender", df["label"])
selected = df[df["label"] == selected_label].iloc[0]
uid = str(selected.get("uid", "")).strip()
selected_payload = selected.drop(labels=["label", "reviewed", "status"], errors="ignore").to_dict()

st.subheader("Tender details")
st.json(selected.drop(labels=["label"]).to_dict())

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Approve (include)"):
        ui.mark_reviewed(uid, status="approved")
        ui.append_row_to_latest_csv("flagged_*.csv", selected_payload)
        ui.remove_uid_from_latest_csv("near_miss_*.csv", uid)
        st.success("Marked as approved")
        st.rerun()
with col2:
    if st.button("Dismiss"):
        ui.mark_reviewed(uid, status="dismissed")
        ui.remove_uid_from_latest_csv("near_miss_*.csv", uid)
        st.success("Marked as dismissed")
        st.rerun()
with col3:
    st.write("Reviewed" if uid in reviewed else "Not reviewed")

hits_raw = selected.get("_hits", "")
if hits_raw:
    st.subheader("Why matched (hits)")
    try:
        st.json(json.loads(hits_raw))
    except json.JSONDecodeError:
        st.write(hits_raw)
