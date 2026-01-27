import html
import smtplib
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from sap_tender_bot.config import NotificationsConfig

EVENT_LABELS = {
    "new": "New tender",
    "amendment": "Amendment posted",
    "close_date_changed": "Close date changed",
    "new_attachment": "New attachment",
}

def _llm_bullets(t):
    llm = t.get("llm") or {}
    bullets = []
    summary = llm.get("summary") or ""
    if summary:
        bullets.append(f"Brief: {summary}")
    attachment_reqs = t.get("attachment_requirements") or []
    for req in attachment_reqs[:3]:
        text = req.get("requirement") if isinstance(req, dict) else str(req)
        if text:
            bullets.append(f"Req (doc): {text}")
    for b in llm.get("key_requirements", [])[:3]:
        if b:
            bullets.append(f"Requirement: {b}")
    for b in llm.get("competitive_signals", [])[:2]:
        if b:
            bullets.append(f"Competitive: {b}")
    next_step = llm.get("recommended_next_step")
    if next_step:
        bullets.append(f"Next step: {next_step}")
    return bullets


def _rationale_bullets(t):
    rationale = t.get("rationale")
    if rationale:
        return [str(rationale)]
    return []


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _event_labels(events) -> str:
    if isinstance(events, str):
        return events
    if not events:
        return ""
    return ", ".join(EVENT_LABELS.get(e, str(e)) for e in events)


def _render_flagged_rows(items, *, include_events: bool = False) -> str:
    rows = []
    for t in items:
        score = t.get("score", 0)
        title = _esc(t.get("title", "Untitled"))
        org = _esc(t.get("org", "Unknown org"))
        close_date = _esc(t.get("close_date", "n/a"))
        url = _esc(t.get("url", ""))
        source = _esc(t.get("source", "source"))
        bullets = _rationale_bullets(t) + _llm_bullets(t)
        bullet_html = "".join(f"<li>{_esc(b)}</li>" for b in bullets) if bullets else ""
        bullets_block = f"<ul>{bullet_html}</ul>" if bullet_html else ""
        events_cell = ""
        if include_events:
            events_cell = f"<td>{_esc(_event_labels(t.get('_events') or t.get('events')))}</td>"

        rows.append(
            """
        <tr>
          <td>{score}</td>
          <td>
            <a href="{url}">{title}</a><br>
            <small>{org} | {source}</small>
            {bullets}
          </td>
          <td>{close_date}</td>
          {events}
        </tr>
        """.format(
                score=score,
                title=title,
                url=url,
                org=org,
                source=source,
                close_date=close_date,
                bullets=bullets_block,
                events=events_cell,
            )
        )
    return "".join(rows)


def _render_watchlist_rows(items) -> str:
    rows = []
    for t in items:
        title = _esc(t.get("title", "Untitled"))
        org = _esc(t.get("org", "Unknown org"))
        close_date = _esc(t.get("close_date", "n/a"))
        url = _esc(t.get("url", ""))
        reason = _esc(t.get("_reject_reason", ""))
        source = _esc(t.get("source", "source"))
        events = _esc(_event_labels(t.get("_events") or t.get("events")))
        rows.append(
            """
        <tr>
          <td>
            <a href="{url}">{title}</a><br>
            <small>{org} | {source}</small>
          </td>
          <td>{close_date}</td>
          <td>{events}</td>
          <td>{reason}</td>
        </tr>
        """.format(
                title=title,
                url=url,
                org=org,
                source=source,
                close_date=close_date,
                events=events,
                reason=reason,
            )
        )
    return "".join(rows)


def _render_upcoming_rows(items) -> str:
    rows = []
    now = datetime.now(timezone.utc)
    for t in items:
        title = _esc(t.get("title", "Untitled"))
        org = _esc(t.get("org", "Unknown org"))
        close_date_raw = t.get("close_date")
        close_date = _esc(close_date_raw or "n/a")
        url = _esc(t.get("url", ""))
        source = _esc(t.get("source", "source"))
        days_left = ""
        if close_date_raw:
            try:
                close_dt = datetime.fromisoformat(str(close_date_raw).replace("Z", "+00:00"))
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)
                days_left = str((close_dt - now).days)
            except Exception:
                days_left = ""
        rows.append(
            """
        <tr>
          <td>
            <a href="{url}">{title}</a><br>
            <small>{org} | {source}</small>
          </td>
          <td>{close_date}</td>
          <td>{days_left}</td>
        </tr>
        """.format(
                title=title,
                url=url,
                org=org,
                source=source,
                close_date=close_date,
                days_left=_esc(days_left),
            )
        )
    return "".join(rows)


def render_digest_html(
    new_items,
    updated_items,
    all_flagged,
    csv_path=None,
    watchlist=None,
    upcoming=None,
    attachment_coverage=None,
):
    new_rows = _render_flagged_rows(new_items or [], include_events=False)
    updated_rows = _render_flagged_rows(updated_items or [], include_events=True)

    new_section = ""
    if new_rows:
        new_section = """
        <h3>High confidence (New)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Score</th>
              <th>Tender</th>
              <th>Closes</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows=new_rows)

    updated_section = ""
    if updated_rows:
        updated_section = """
        <h3>High confidence (Updated/Amended)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Score</th>
              <th>Tender</th>
              <th>Closes</th>
              <th>Update</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows=updated_rows)

    watchlist = watchlist or []
    watch_section = ""
    if watchlist:
        watch_rows = _render_watchlist_rows(watchlist)
        watch_section = """
        <h3>Watchlist (near-miss)</h3>
        <p>Items rejected for a single reason.</p>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Tender</th>
              <th>Closes</th>
              <th>Event</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows=watch_rows)

    upcoming = upcoming or []
    upcoming_section = ""
    if upcoming:
        upcoming_rows = _render_upcoming_rows(upcoming)
        upcoming_section = """
        <h3>Upcoming closes (next 7 days)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Tender</th>
              <th>Closes</th>
              <th>Days left</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows=upcoming_rows)

    today = datetime.now().strftime("%Y-%m-%d")
    csv_note = f"<p>CSV exported to: <code>{_esc(csv_path)}</code></p>" if csv_path else ""
    total_top = len(new_items or []) + len(updated_items or [])
    coverage_note = ""
    if attachment_coverage and attachment_coverage.get("total", 0):
        with_reqs = int(attachment_coverage.get("with_requirements", 0))
        total = int(attachment_coverage.get("total", 0))
        pct = int(round((with_reqs / total) * 100)) if total > 0 else 0
        coverage_note = (
            f"<p>Attachment coverage: <b>{pct}%</b> ({with_reqs} of {total})</p>"
        )
    return """
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2>SAP-Relevant Public Sector Tenders Digest - {today}</h2>
        <p>High-confidence items: <b>{top_count}</b> (of {all_count})</p>
        {coverage_note}
        {csv_note}
        {new_section}
        {updated_section}
        {upcoming_section}
        {watch_section}
      </body>
    </html>
    """.format(
        today=today,
        top_count=total_top,
        all_count=len(all_flagged),
        coverage_note=coverage_note,
        csv_note=csv_note,
        new_section=new_section,
        updated_section=updated_section,
        watch_section=watch_section,
        upcoming_section=upcoming_section,
    )



def _attach_csv(msg, csv_path: Path):
    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_path.read_bytes())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
    msg.attach(part)



def send_digest_email(
    new_items,
    updated_items,
    all_flagged=None,
    csv_path=None,
    watchlist=None,
    upcoming=None,
    attachment_coverage=None,
    notifications: NotificationsConfig | None = None,
):
    if notifications is None:
        raise ValueError("notifications config is required to send email")

    smtp_host = notifications.smtp_host
    smtp_port = notifications.smtp_port
    smtp_user = notifications.smtp_user
    smtp_pass = notifications.smtp_pass
    recipients = notifications.email_recipients
    email_from = notifications.email_from or smtp_user

    if not smtp_host or not smtp_user or not smtp_pass or not recipients:
        raise ValueError("SMTP configuration or recipients missing")

    all_flagged = all_flagged or list(new_items or []) + list(updated_items or [])

    msg = MIMEMultipart("alternative")
    total_top = len(new_items or []) + len(updated_items or [])
    msg["Subject"] = f"SAP Tender Digest ({total_top} of {len(all_flagged)} flagged)"
    msg["From"] = email_from
    msg["To"] = ", ".join(recipients)

    html = render_digest_html(
        new_items,
        updated_items,
        all_flagged,
        csv_path=csv_path,
        watchlist=watchlist,
        upcoming=upcoming,
        attachment_coverage=attachment_coverage,
    )
    msg.attach(MIMEText(html, "html"))

    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            _attach_csv(msg, csv_path)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(email_from, recipients, msg.as_string())
