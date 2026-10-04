"""Tell a person something about a run: always in the log, by email when set up.

A corpus run takes days, and nobody watches a log for days. `notify` logs
every message, and also emails it through Gmail's SMTP server when `.env`
holds the three mail settings -- the sending address, its app password
(Google account -> Security -> App passwords) and the recipient. Without them
it only logs, so a test or a notebook needs no setup.

The caller decides when to notify; a run calls it at most three times. A mail
that cannot be sent is logged and dropped: the run must not die of its alarm.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from engine import config

log = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465  # SSL from the first byte
_SMTP_TIMEOUT_S = 30


def notify(subject: str, body: str) -> None:
    """Log `subject` and `body`; email them too when the mail settings are in `.env`."""
    log.info("%s\n%s", subject, body)

    sender = config.get("NOTIFY_GMAIL_ADDRESS")
    password = config.get("NOTIFY_GMAIL_APP_PASSWORD")
    recipient = config.get("NOTIFY_TO")
    if not (sender and password and recipient):
        return

    message = EmailMessage()
    message["Subject"] = f"[supplement-kg] {subject}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)
    try:
        with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT, timeout=_SMTP_TIMEOUT_S) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        log.warning("could not email %r to %s: %r", subject, recipient, exc)
