import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _html_digest(tenders):
    rows = []
    for t in tenders:
        score = t.get("score", 0)
        title = t.get("title", "Untitled")
        org = t.get("org", "Unknown org")
        close_date = t.get("close_date", "n/a")
        url = t.get("url", "")
        source = t.get("source", "source")

        rows.append(f"""
        <tr>
          <td>{score}</td>
          <td><a href="{url}">{title}</a><br><small>{org} • {source}</small></td>
          <td>{close_date}</td>
        </tr>
        """)

    today = datetime.now().strftime("%Y-%m-%d")
    return f"""
    <html>
      <body>
        <h2>SAP-Relevant Public Sector Tenders Digest — {today}</h2>
        <p>Flagged tenders: <b>{len(tenders)}</b></p>
        <table border="1" cellpadding="8" cellspacing="0">
          <thead>
            <tr>
              <th>Score</th>
              <th>Tender</th>
              <th>Closes</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </body>
    </html>
    """


def send_digest_email(flagged_tenders):
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]
    email_to = os.environ["EMAIL_TO"]
    email_from = os.environ.get("EMAIL_FROM", smtp_user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"SAP Tender Digest ({len(flagged_tenders)} flagged)"
    msg["From"] = email_from
    msg["To"] = email_to

    html = _html_digest(flagged_tenders)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(email_from, [email_to], msg.as_string())
