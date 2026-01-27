from __future__ import annotations

import json

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()

st.title("Attachments")

df = ui.load_export_df("flagged_*.csv")
df = ui.apply_filters(df, filters)

if df.empty:
    st.info("No flagged tenders found. Run a digest export to populate data.")
    st.stop()

df = df.reset_index(drop=True)
df["label"] = df.apply(
    lambda row: " | ".join(
        [
            str(row.get("title", "")).strip() or "Untitled",
            str(row.get("org", "")).strip() or "Unknown org",
            str(row.get("tender_key", "")).strip() or str(row.get("uid", "")).strip(),
        ]
    ),
    axis=1,
)

selected_label = st.selectbox("Select a tender", df["label"])
selected = df[df["label"] == selected_label].iloc[0]

tender_key = str(selected.get("tender_key", "")).strip()
uid = str(selected.get("uid", "")).strip()
payload = ui.load_tender_payload(tender_key=tender_key, uid=uid)

st.subheader("Tender overview")
st.json(selected.drop(labels=["label"]).to_dict())

if not payload:
    st.warning(
        "No stored payload found for this tender. Attachment summaries are persisted on non-dry-run runs."
    )
    st.stop()

attachments = payload.get("attachment_summaries") or []
attachment_reqs = payload.get("attachment_requirements") or []

st.subheader("Attachment requirements")
if attachment_reqs:
    for req in attachment_reqs:
        if isinstance(req, dict):
            text = req.get("requirement") or ""
            name = req.get("attachment_name") or req.get("attachment_url") or ""
            st.markdown(f"- **{text}**  \n  _{name}_")
        else:
            st.markdown(f"- {req}")
else:
    st.caption("No attachment-derived requirements found.")

st.subheader("Attachment summaries")
if not attachments:
    st.caption("No attachment summaries stored for this tender.")
else:
    for idx, summary in enumerate(attachments, start=1):
        if not isinstance(summary, dict):
            continue
        name = summary.get("attachment_name") or summary.get("attachment_url") or f"Attachment {idx}"
        with st.expander(name, expanded=False):
            url = summary.get("attachment_url")
            if url:
                st.markdown(f"[Open attachment]({url})")
            st.caption(
                f"MIME: {summary.get('mime_type') or 'n/a'} | "
                f"Chars: {summary.get('text_chars') or 'n/a'} | "
                f"Bytes: {summary.get('size_bytes') or 'n/a'}"
            )
            memo = summary.get("memo")
            if memo:
                st.write(f"Memo: {memo}")
            sections = summary.get("source_sections") or []
            if sections:
                st.write("Sections captured:")
                st.write(", ".join(sections))
            st.write("Summary JSON:")
            st.json(summary.get("summary") or {})

st.subheader("Stored payload (raw)")
st.json(payload if isinstance(payload, dict) else json.loads(payload))

st.subheader("Tender Q&A (idea)")
st.info(
    "Next step could be a Q&A panel that indexes attachment text and supports questions "
    "with citations back to specific sections. Happy to prototype this when you're ready."
)
