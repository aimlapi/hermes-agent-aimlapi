import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from providers import get_provider_profile

assert get_provider_profile("aimlapi") is not None

from plugins.model_providers.aimlapi.client import APIError, AimlapiClient
from plugins.model_providers.aimlapi.config import DEFAULT_PARTNER_ID, Endpoints


class _Handler(BaseHTTPRequestHandler):
    request_headers = None

    def do_POST(self):
        type(self).request_headers = self.headers
        body = json.dumps({"key": "issued-key"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def _trusted_client():
    return AimlapiClient(
        Endpoints(
            "https://auth.aimlapi.com",
            "https://app.aimlapi.com",
            "https://api.aimlapi.com/v1",
            "https://pay.aimlapi.com",
            "https://aimlapi.com/app",
        )
    )


def test_client_sends_bearer_and_attribution_headers():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    client = AimlapiClient(
        Endpoints(base, base, base, base, base),
        allow_untrusted_test_endpoints=True,
    )
    try:
        key = client.create_key("secret-key")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert key == "issued-key"
    headers = _Handler.request_headers
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["X-AIMLAPI-Source"] == "agent"
    assert headers["X-AIMLAPI-Partner-ID"] == DEFAULT_PARTNER_ID
    assert headers["X-AIMLAPI-Integration-Repo"] == "NousResearch/hermes-agent"
    assert headers["X-AIMLAPI-Integration-Version"] == "1.0.0"


def test_create_checkout_session_uses_app_return_url(monkeypatch):
    client = _trusted_client()
    captured = {}

    def request(method, url, *, bearer="", body=None):
        captured.update(method=method, url=url, bearer=bearer, body=body)
        return {"sessionToken": "session"}

    monkeypatch.setattr(client, "_request", request)

    assert client.create_checkout_session() == "session"
    assert captured["body"]["returnUrl"] == "https://aimlapi.com/app"
    assert captured["body"]["partnerId"] == DEFAULT_PARTNER_ID


def test_passwordless_signup_omits_invite_code_by_default(monkeypatch):
    client = _trusted_client()
    captured = {}
    monkeypatch.delenv("AIMLAPI_INVITE_CODE", raising=False)

    def request(method, url, *, bearer="", body=None):
        captured.update(method=method, url=url, bearer=bearer, body=body)
        return {"token": "session-token"}

    monkeypatch.setattr(client, "_request", request)

    assert client.create_passwordless_account("user@example.com") == "session-token"
    assert captured["body"] == {"email": "user@example.com"}


def test_passwordless_signup_sends_staging_invite_code(monkeypatch):
    client = _trusted_client()
    captured = {}
    monkeypatch.setenv("AIMLAPI_INVITE_CODE", "staging-invite")

    def request(method, url, *, bearer="", body=None):
        captured.update(method=method, url=url, bearer=bearer, body=body)
        return {"token": "session-token"}

    monkeypatch.setattr(client, "_request", request)

    assert client.create_passwordless_account("user@aimlapi.com") == "session-token"
    assert captured["body"] == {
        "email": "user@aimlapi.com",
        "inviteCode": "staging-invite",
    }


def test_checkout_poll_retries_transient_failure(monkeypatch):
    client = _trusted_client()
    outcomes = iter((APIError("temporary", status=503), {"status": "paid"}))

    def get_session(_token):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client, "get_checkout_session", get_session)

    assert client.wait_for_checkout("session", poll_interval=0) == "paid"


@pytest.mark.parametrize("status", ("cancelled", "expired", "failed"))
def test_checkout_poll_surfaces_terminal_state(monkeypatch, status):
    client = _trusted_client()
    monkeypatch.setattr(
        client,
        "get_checkout_session",
        lambda _token: {"status": status},
    )

    with pytest.raises(APIError, match=status):
        client.wait_for_checkout("session", poll_interval=0)


def test_checkout_poll_times_out_without_request(monkeypatch):
    client = _trusted_client()
    monkeypatch.setattr(
        client,
        "get_checkout_session",
        lambda _token: pytest.fail("expired deadline must not poll"),
    )

    with pytest.raises(APIError, match="timed out"):
        client.wait_for_checkout("session", timeout=0, poll_interval=0)


@pytest.mark.parametrize(
    ("auto_top_up", "expected"),
    ((True, True), (False, None)),
)
def test_checkout_only_sends_auto_top_up_when_enabled(
    monkeypatch, auto_top_up, expected
):
    client = _trusted_client()
    captured = {}

    def request(method, url, *, bearer="", body=None):
        captured.update(method=method, url=url, bearer=bearer, body=body)
        return {"checkout": {"payUrl": "https://pay.aimlapi.com/checkout"}}

    monkeypatch.setattr(client, "_request", request)

    client.start_checkout(
        "auth-token",
        "session",
        2500,
        "payment-id",
        auto_top_up,
    )

    assert captured["body"].get("autoTopUp") is expected


def test_checkout_rejects_non_aimlapi_redirect(monkeypatch):
    client = _trusted_client()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "checkout": {"payUrl": "https://attacker.example/checkout"}
        },
    )

    with pytest.raises(APIError, match="invalid checkout URL"):
        client.start_checkout(
            "auth-token",
            "session",
            2500,
            "payment-id",
            True,
        )
