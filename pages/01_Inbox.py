from __future__ import annotations

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()

st.title("Inbox")

df = ui.load_export_df("flagged_*.csv")
df = ui.apply_filters(df, filters)

ui_state = ui.load_ui_state()
reviewed = set(ui_state.get("reviewed_uids") or [])

if "uid" in df.columns:
    df["reviewed"] = df["uid"].isin(reviewed)

if df.empty:
    st.info("No flagged tenders found. Run a digest export to populate data.")
    st.stop()

display_cols = [c for c in ["reviewed", "title", "org", "publish_date", "close_date", "score", "url"] if c in df.columns]
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
st.subheader("Tender details")
st.json(selected.drop(labels=["label"]).to_dict())

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Mark reviewed"):
        ui.mark_reviewed(uid, status="reviewed")
        st.success("Marked as reviewed")
with col2:
    st.write("Reviewed" if uid in reviewed else "Not reviewed")

note_key = f"note_{uid}"
note = ui_state.get("notes", {}).get(uid, "")
note = st.text_area("Notes", value=note, key=note_key, height=120)
if st.button("Save note"):
    ui.save_note(uid, note.strip())
    st.success("Note saved")
