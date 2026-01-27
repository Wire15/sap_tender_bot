from __future__ import annotations

import streamlit as st

import ui_common as ui

ui.set_page_config("SAP Tender Admin")
filters = ui.render_sidebar()
role = filters.get("role")

st.title("Settings")

config = ui.load_ui_config()
notifications = config.notifications

st.subheader("Notifications")
disabled = role != "Admin"
if disabled:
    st.warning("Admin access required to edit settings.")

email_recipients = st.text_area(
    "Email recipients (comma-separated)",
    value=",".join(notifications.email_recipients),
    disabled=disabled,
)
slack_webhook = st.text_input(
    "Slack webhook URL",
    value=notifications.slack_webhook,
    disabled=disabled,
)
digest_schedule = st.text_input(
    "Digest schedule",
    value=notifications.digest_schedule,
    disabled=disabled,
)

if st.button("Save notifications", disabled=disabled):
    config.notifications.email_recipients = [
        v.strip() for v in email_recipients.split(",") if v.strip()
    ]
    config.notifications.slack_webhook = slack_webhook.strip()
    config.notifications.digest_schedule = digest_schedule.strip()
    ui.save_ui_config(config)
    st.success("Notifications saved")

st.subheader("Latest exports")
exports = [
    ("Flagged CSV", ui.latest_export("flagged_*.csv")),
    ("Near-miss CSV", ui.latest_export("near_miss_*.csv")),
    ("Reject report", ui.latest_export("reject_report_*.json")),
]
for label, path in exports:
    if path and path.exists():
        st.write(f"{label}: {path.name}")
        with path.open("rb") as handle:
            st.download_button(f"Download {label}", data=handle.read(), file_name=path.name)
    else:
        st.write(f"{label}: n/a")

st.subheader("Credentials status")
status = ui.env_status()
rows = [{"Variable": key, "Present": "yes" if value else "no"} for key, value in status.items()]
st.dataframe(rows, use_container_width=True, hide_index=True)
