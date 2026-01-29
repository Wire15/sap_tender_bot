from __future__ import annotations

import json
import random

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()

st.title("Attachments")

coverage_report = ui.load_attachment_coverage_report()
if coverage_report:
    st.subheader("Attachment coverage (last 30 days)")
    baseline = coverage_report.get("baseline", {})
    rates = baseline.get("rates", {})
    st.write(
        {
            "tenders": baseline.get("tenders"),
            "summaries_rate": rates.get("summaries"),
            "requirements_rate": rates.get("requirements"),
            "dates_rate": rates.get("dates"),
            "evaluation_rate": rates.get("evaluation"),
            "submission_rate": rates.get("submission"),
            "low_signal_rate": baseline.get("low_signal_rate"),
        }
    )
    weekly_delta = coverage_report.get("weekly_delta", {})
    if weekly_delta:
        st.caption(
            "Weekly delta (pp): "
            f"req {weekly_delta.get('requirements', 0)} | "
            f"dates {weekly_delta.get('dates', 0)} | "
            f"eval {weekly_delta.get('evaluation', 0)} | "
            f"submission {weekly_delta.get('submission', 0)} | "
            f"summaries {weekly_delta.get('summaries', 0)}"
        )

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
            structured = summary.get("structured_summary")
            if not structured and isinstance(summary.get("summary"), dict):
                structured = summary["summary"].get("structured_summary")
            if structured:
                st.write("Structured summary:")
                for category, items in structured.items():
                    if not items:
                        continue
                    st.markdown(f"**{category}**")
                    for item in items:
                        if isinstance(item, dict):
                            text = item.get("text") or ""
                            citations = item.get("citations") or []
                        else:
                            text = str(item)
                            citations = []
                        if text:
                            st.markdown(f"- {text}")
                        if citations and isinstance(citations, list):
                            first = citations[0] if citations else {}
                            section = (
                                first.get("section_heading") if isinstance(first, dict) else None
                            )
                            page = first.get("page") if isinstance(first, dict) else None
                            section = section or "n/a"
                            page_label = f"p.{page}" if page is not None else "p.?"
                            st.caption(f"Provenance: {name} | {section} | {page_label}")
                        else:
                            st.caption("Provenance: unavailable")

st.subheader("Stored payload (raw)")
st.json(payload if isinstance(payload, dict) else json.loads(payload))

st.subheader("Manual evidence-quality spot checks")
st.caption("Sample 20 attachment summaries for manual review.")
if st.button("Sample 20 summaries"):
    samples = []
    for _, row in df.head(50).iterrows():
        tender_key = str(row.get("tender_key", "")).strip()
        uid = str(row.get("uid", "")).strip()
        payload_row = ui.load_tender_payload(tender_key=tender_key, uid=uid)
        if not payload_row:
            continue
        for summary in payload_row.get("attachment_summaries") or []:
            if not isinstance(summary, dict):
                continue
            samples.append(
                {
                    "title": row.get("title"),
                    "org": row.get("org"),
                    "attachment": summary.get("attachment_name") or summary.get("attachment_url"),
                    "memo": summary.get("memo"),
                    "requirements": summary.get("summary", {}).get("requirements", []),
                    "dates": summary.get("summary", {}).get("dates", []),
                }
            )
        if len(samples) >= 40:
            break
    if samples:
        random.shuffle(samples)
        st.session_state["spot_check_samples"] = samples[:20]
    else:
        st.session_state["spot_check_samples"] = []

samples = st.session_state.get("spot_check_samples") or []
if samples:
    for item in samples:
        st.markdown(
            f"**{item.get('title', 'Untitled')}**  \n"
            f"_{item.get('org', 'Unknown org')}_"
        )
        st.caption(f"Attachment: {item.get('attachment')}")
        memo = item.get("memo") or ""
        if memo:
            st.write(f"Memo: {memo}")
        reqs = item.get("requirements") or []
        dates = item.get("dates") or []
        if reqs:
            st.write("Reqs: " + ", ".join(str(r) for r in reqs[:2]))
        if dates:
            st.write("Dates: " + ", ".join(str(d) for d in dates[:2]))
        st.divider()

st.subheader("Tender Q&A (idea)")
st.info(
    "Next step could be a Q&A panel that indexes attachment text and supports questions "
    "with citations back to specific sections. Happy to prototype this when you're ready."
)
