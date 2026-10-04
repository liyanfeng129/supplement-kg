"""notify: always logs; emails through Gmail SMTP only when `.env` says how.

What is pinned: every call logs subject and body; without the three mail
settings nothing is sent; with them one message goes to the recipient through
`smtp.gmail.com` over SSL, logged in with the app password; and a mail that
fails to send is logged, never raised -- a run must not die of its alarm.

No network: `smtplib.SMTP_SSL` is replaced.
"""

from __future__ import annotations

import logging
import smtplib

import pytest

from engine import notify as notify_module
from engine.notify import notify

MAIL_KEYS = ("NOTIFY_GMAIL_ADDRESS", "NOTIFY_GMAIL_APP_PASSWORD", "NOTIFY_TO")


class FakeSMTP:
    sent: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.logins = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, message):
        FakeSMTP.sent.append((self, message))


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(notify_module.smtplib, "SMTP_SSL", FakeSMTP)
    for key in MAIL_KEYS:
        monkeypatch.delenv(key, raising=False)


def with_mail(monkeypatch):
    monkeypatch.setenv("NOTIFY_GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("NOTIFY_GMAIL_APP_PASSWORD", "abcd efgh")
    monkeypatch.setenv("NOTIFY_TO", "you@example.com")


def test_without_mail_settings_it_only_logs(caplog):
    with caplog.at_level(logging.INFO):
        notify("run finished", "88000 ok")

    assert "run finished" in caplog.text and "88000 ok" in caplog.text
    assert FakeSMTP.sent == []


def test_with_mail_settings_it_emails_the_recipient(monkeypatch, caplog):
    with_mail(monkeypatch)

    with caplog.at_level(logging.INFO):
        notify("run finished", "88000 ok")

    [(smtp, message)] = FakeSMTP.sent
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 465)
    assert smtp.logins == [("me@gmail.com", "abcd efgh")]
    assert message["To"] == "you@example.com"
    assert message["From"] == "me@gmail.com"
    assert "run finished" in message["Subject"]
    assert "88000 ok" in message.get_content()
    assert "run finished" in caplog.text


def test_a_partial_mail_setting_sends_nothing(monkeypatch):
    monkeypatch.setenv("NOTIFY_GMAIL_ADDRESS", "me@gmail.com")

    notify("run finished", "body")

    assert FakeSMTP.sent == []


def test_a_mail_that_fails_to_send_is_logged_not_raised(monkeypatch, caplog):
    with_mail(monkeypatch)

    def refuse(self, user, password):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(FakeSMTP, "login", refuse)

    with caplog.at_level(logging.WARNING):
        notify("model unavailable", "body")

    assert "could not email" in caplog.text
