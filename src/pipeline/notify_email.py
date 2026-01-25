import html
import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def _llm_bullets(t):
    llm = t.get("llm") or {}
    bullets = []
    for b in llm.get("reasoning_bullets", [])[:6]:
        if b:
            bullets.append(str(b))
    for b in llm.get("red_flags", [])[:3]:
        if b:
            bullets.append(f"Red flag: {b}")
    return bullets


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def render_digest_html(flagged_top, all_flagged, csv_path=None, watchlist=None):
    rows = []
    for t in flagged_top:
        score = t.get("score", 0)
        title = _esc(t.get("title", "Untitled"))
        org = _esc(t.get("org", "Unknown org"))
        close_date = _esc(t.get("close_date", "n/a"))
        url = _esc(t.get("url", ""))
        source = _esc(t.get("source", "source"))
        bullets = _llm_bullets(t)
        bullet_html = "".join(f"<li>{_esc(b)}</li>" for b in bullets) if bullets else ""
        bullets_block = f"<ul>{bullet_html}</ul>" if bullet_html else ""

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
        </tr>
        """.format(
                score=score,
                title=title,
                url=url,
                org=org,
                source=source,
                close_date=close_date,
                bullets=bullets_block,
            )
        )

    watchlist = watchlist or []
    watch_rows = []
    for t in watchlist:
        title = _esc(t.get("title", "Untitled"))
        org = _esc(t.get("org", "Unknown org"))
        close_date = _esc(t.get("close_date", "n/a"))
        url = _esc(t.get("url", ""))
        reason = _esc(t.get("_reject_reason", ""))
        source = _esc(t.get("source", "source"))
        watch_rows.append(
            """
        <tr>
          <td>
            <a href="{url}">{title}</a><br>
            <small>{org} | {source}</small>
          </td>
          <td>{close_date}</td>
          <td>{reason}</td>
        </tr>
        """.format(
                title=title,
                url=url,
                org=org,
                source=source,
                close_date=close_date,
                reason=reason,
            )
        )

    watch_section = ""
    if watch_rows:
        watch_section = """
        <h3>Watchlist (near-miss)</h3>
        <p>Items rejected for a single reason.</p>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Tender</th>
              <th>Closes</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(
            rows="".join(watch_rows),
        )

    today = datetime.now().strftime("%Y-%m-%d")
    csv_note = f"<p>CSV exported to: <code>{_esc(csv_path)}</code></p>" if csv_path else ""
    return """
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2>SAP-Relevant Public Sector Tenders Digest - {today}</h2>
        <p>New flagged tenders: <b>{top_count}</b> (of {all_count})</p>
        {csv_note}
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
        {watch_section}
      </body>
    </html>
    """.format(
        today=today,
        top_count=len(flagged_top),
        all_count=len(all_flagged),
        rows="".join(rows),
        csv_note=csv_note,
        watch_section=watch_section,
    )



def _attach_csv(msg, csv_path: Path):
    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_path.read_bytes())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
    msg.attach(part)



def send_digest_email(flagged_top, all_flagged=None, csv_path=None, watchlist=None):
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]
    email_to = os.environ["EMAIL_TO"]
    email_from = os.environ.get("EMAIL_FROM", smtp_user)

    all_flagged = all_flagged or flagged_top

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"SAP Tender Digest ({len(flagged_top)} of {len(all_flagged)} flagged)"
    msg["From"] = email_from
    msg["To"] = email_to

    html = render_digest_html(flagged_top, all_flagged, csv_path=csv_path, watchlist=watchlist)
    msg.attach(MIMEText(html, "html"))

    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            _attach_csv(msg, csv_path)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(email_from, [email_to], msg.as_string())
