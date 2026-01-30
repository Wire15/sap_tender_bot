import html
import re
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

_SUBMISSION_REQ_TERMS = {
    "submission",
    "submit",
    "bid",
    "response format",
    "proposal format",
    "file size",
    "portal",
    "email",
    "deadline",
    "due date",
    "closing date",
    "must be submitted",
    "submit via",
    "submit by",
    "response must",
    "proposal must",
}

_TECHNICAL_REQ_TERMS = {
    "integration",
    "api",
    "interface",
    "architecture",
    "data",
    "database",
    "security",
    "performance",
    "availability",
    "scalability",
    "cloud",
    "migration",
    "modernization",
    "modernisation",
    "implementation",
    "deployment",
    "testing",
    "monitoring",
    "analytics",
    "reporting",
    "governance",
    "compliance",
    "privacy",
    "backup",
    "disaster recovery",
    "resilience",
    "encryption",
    "authentication",
    "authorization",
    "identity",
    "access",
    "network",
    "hosting",
    "storage",
    "etl",
    "integration platform",
    "sensing",
    "risk",
    "mapping",
    "modernization plan",
}

_STAFFING_REQ_TERMS = {
    "level",
    "resource",
    "resources",
    "consultant",
    "consultants",
    "specialist",
    "analyst",
    "architect",
    "project manager",
    "project executive",
    "developer",
    "engineer",
    "tbips",
    "spict",
}

_SUBMISSION_BLOCKLIST_TERMS = {
    "level",
    "resource",
    "resources",
    "analyst",
    "architect",
    "project manager",
    "project executive",
    "tbips",
    "spict",
}

_SAP_SIGNAL_TERMS = {
    "sap",
    "s/4",
    "s4",
    "s/4hana",
    "hana",
    "ecc",
    "erp",
    "ariba",
    "successfactors",
}


def _has_sap_signal(t) -> bool:
    parts = [t.get("title", ""), t.get("org", ""), t.get("rationale", "")]
    llm = t.get("llm") or {}
    parts.append(llm.get("summary", ""))
    for req in t.get("attachment_requirements") or []:
        if isinstance(req, dict):
            parts.append(req.get("requirement", ""))
        else:
            parts.append(str(req))
    for summary in t.get("attachment_summaries") or []:
        if isinstance(summary, dict):
            parts.extend(summary.get("summary", {}).get("requirements", []))
    text = " ".join(str(p) for p in parts if p).lower()
    if any(term in text for term in _SAP_SIGNAL_TERMS):
        return True
    bucket = str(t.get("semantic_bucket") or "").lower()
    return bucket in {"erp modernization", "procurement", "hr/payroll", "integration/data", "ams"}


def _attachment_note_lines(t):
    lines = []
    for summary in t.get("attachment_summaries") or []:
        if not isinstance(summary, dict):
            continue
        if summary.get("low_signal"):
            continue
        name = summary.get("attachment_name") or "Attachment"
        payload = summary.get("summary") or {}
        reqs = payload.get("requirements") or []
        dates = payload.get("dates") or []
        parts = []
        if reqs:
            parts.append("Reqs: " + ", ".join(str(r) for r in reqs[:2]))
        if dates:
            parts.append("Dates: " + ", ".join(str(d) for d in dates[:2]))
        if parts:
            line = f"{name}: " + " | ".join(parts)
            if len(line) > 200:
                line = (line[:200].rsplit(" ", 1)[0].strip()) or line[:200]
            lines.append(line)
        if len(lines) >= 2:
            break
    return lines


def _llm_bullets(t):
    llm = t.get("llm") or {}
    bullets = []
    summary = llm.get("summary") or ""
    if summary:
        bullets.append(f"Brief: {summary}")
    signal = _attachment_signal(t)
    if signal:
        bullets.append(signal)
    for line in _attachment_note_lines(t):
        bullets.append(line)
    if _has_sap_signal(t):
        bullets.append("Why it matters: SAP/ERP signal found.")
    attachment_reqs = t.get("attachment_requirements") or []
    submission_reqs = []
    technical_reqs = []
    staffing_reqs = []
    for req in attachment_reqs:
        text = req.get("requirement") if isinstance(req, dict) else str(req)
        if not text:
            continue
        lower = text.lower()
        if any(term in lower for term in _STAFFING_REQ_TERMS):
            # drop staffing-like requirements from the digest entirely
            staffing_reqs.append(text)
            continue
        if any(term in lower for term in _SUBMISSION_BLOCKLIST_TERMS):
            # prevent role-level items from being labeled as submission
            continue
        is_submission = any(term in lower for term in _SUBMISSION_REQ_TERMS)
        is_technical = any(term in lower for term in _TECHNICAL_REQ_TERMS)
        is_staffing = any(term in lower for term in _STAFFING_REQ_TERMS)
        if is_technical and not is_submission:
            technical_reqs.append(text)
        elif is_submission and not is_technical:
            submission_reqs.append(text)
        else:
            # Ambiguous: prefer technical if it contains any tech term.
            if is_technical:
                technical_reqs.append(text)
            elif is_staffing:
                staffing_reqs.append(text)
            else:
                # default to technical unless it explicitly looks like submission
                if is_submission:
                    submission_reqs.append(text)
                else:
                    technical_reqs.append(text)
    # De-dup while preserving order.
    submission_reqs = list(dict.fromkeys(submission_reqs))
    technical_reqs = list(dict.fromkeys(technical_reqs))
    staffing_reqs = list(dict.fromkeys(staffing_reqs))
    for text in submission_reqs[:1]:
        bullets.append(f"Req (submission): {text}")
    for text in technical_reqs[:2]:
        bullets.append(f"Req (technical): {text}")
    # Staffing requirements are intentionally omitted from the digest.
    if t.get("attachment_summary_used") and not attachment_reqs:
        for summary in t.get("attachment_summaries") or []:
            if not isinstance(summary, dict):
                continue
            if summary.get("low_signal"):
                continue
            memo = summary.get("memo") or ""
            if memo:
                bullets.append(memo)
                break
    if not t.get("attachment_summary_used"):
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


def _attachment_signal(t) -> str:
    summaries = t.get("attachment_summaries") or []
    requirements = t.get("attachment_requirements") or []
    if not summaries and not requirements:
        return ""
    attachment_names = set()
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("attachment_name"):
            attachment_names.add(str(summary.get("attachment_name")))
    if not attachment_names:
        for req in requirements:
            if isinstance(req, dict) and req.get("attachment_name"):
                attachment_names.add(str(req.get("attachment_name")))

    dates_found = False
    risks_found = False
    structured_found = False
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        payload = summary.get("summary") or {}
        if payload.get("dates"):
            dates_found = True
        if payload.get("key_risks"):
            risks_found = True
        structured = summary.get("structured_summary")
        if not structured and isinstance(payload, dict):
            structured = payload.get("structured_summary")
        if isinstance(structured, dict):
            for items in structured.values():
                if isinstance(items, list) and items:
                    structured_found = True
                    break
        if structured_found:
            break

    attachment_count = len(attachment_names) if attachment_names else len(summaries)
    req_count = len(requirements)
    dates_flag = "yes" if dates_found else "no"
    risks_flag = "yes" if risks_found else "no"
    if req_count == 0 and not dates_found and not risks_found and not structured_found:
        return ""
    return (
        "Attachment signal: "
        f"{attachment_count} attachments, reqs={req_count}, dates={dates_flag}, risks={risks_flag}"
    )


def _rationale_bullets(t):
    rationale = t.get("rationale")
    if rationale:
        return [str(rationale)]
    return []


def _fix_mojibake(text: str) -> str:
    if not text:
        return ""
    suspect = (
        "\u00e2\u20ac\u2122",
        "\u00e2\u20ac\u201c",
        "\u00e2\u20ac",
        "\u00c3",
        "\u00c2",
        "\u00a4",
    )
    if any(token in text for token in suspect):
        try:
            repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                text = repaired
        except Exception:
            pass
    replacements = {
        "\u00e2\u20ac\u2122": "\u2019",
        "\u00e2\u20ac\u201c": "\u201c",
        "\u00e2\u20ac\u009d": "\u201d",
        "\u00e2\u20ac\u0098": "\u2018",
        "\u00e2\u20ac\u0099": "\u2019",
        "\u00e2\u20ac\u2013": "\u2013",
        "\u00e2\u20ac\u2014": "\u2014",
        "\u00c2 ": " ",
        "\u00c2": "",
    }
    for bad, good in replacements.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def _esc(value) -> str:
    if value is None:
        return ""
    cleaned = _fix_mojibake(str(value))
    return html.escape(cleaned, quote=True)


def _sanitize_url(value) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v)
    text = str(value)
    if "," in text:
        text = text.split(",", 1)[0].strip()
    match = re.search(r"https?://\\S+", text)
    if match:
        return match.group(0).rstrip(").,;")
    return text.split()[0] if text.split() else ""


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
        url = _esc(_sanitize_url(t.get("url", "")))
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
        url = _esc(_sanitize_url(t.get("url", "")))
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
        url = _esc(_sanitize_url(t.get("url", "")))
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
    attachment_notes=None,
    attachment_structured=None,
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

    attachment_notes = attachment_notes or []
    notes_section = ""
    if attachment_notes:
        rows = []
        for note in attachment_notes[:15]:
            title = _esc(note.get("title", "Untitled"))
            org = _esc(note.get("org", "Unknown org"))
            url = _esc(_sanitize_url(note.get("url", "")))
            memo = _esc(note.get("memo", ""))
            if not memo:
                continue
            rows.append(
                """
            <tr>
              <td>
                <a href="{url}">{title}</a><br>
                <small>{org}</small>
              </td>
              <td>{memo}</td>
            </tr>
            """.format(title=title, url=url, org=org, memo=memo)
            )
        if rows:
            notes_section = """
        <h3>Attachment notes</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Tender</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows="".join(rows))

    attachment_structured = attachment_structured or []
    structured_section = ""
    if attachment_structured:
        rows = []
        for item in attachment_structured[:30]:
            title = _esc(item.get("title", "Untitled"))
            org = _esc(item.get("org", "Unknown org"))
            url = _esc(_sanitize_url(item.get("url", "")))
            attachment_name = _esc(item.get("attachment_name", "Attachment"))
            category = _esc(item.get("category", ""))
            text = _esc(item.get("text", ""))
            provenance = _esc(item.get("provenance", ""))
            if not text:
                continue
            rows.append(
                """
            <tr>
              <td>
                <a href="{url}">{title}</a><br>
                <small>{org}</small>
              </td>
              <td>{attachment_name}</td>
              <td>{category}</td>
              <td>{text}<br><small>{provenance}</small></td>
            </tr>
            """.format(
                    title=title,
                    url=url,
                    org=org,
                    attachment_name=attachment_name,
                    category=category,
                    text=text,
                    provenance=provenance,
                )
            )
        if rows:
            structured_section = """
        <h3>Attachment structured summary</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Tender</th>
              <th>Attachment</th>
              <th>Category</th>
              <th>Item</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(rows="".join(rows))

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
        {notes_section}
        {structured_section}
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
        notes_section=notes_section,
        structured_section=structured_section,
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
    attachment_notes=None,
    attachment_structured=None,
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
        attachment_notes=attachment_notes,
        attachment_structured=attachment_structured,
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
