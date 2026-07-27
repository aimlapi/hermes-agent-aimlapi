"""Configuration and attribution for the AI/ML API Hermes plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote, urlparse

DEFAULT_PARTNER_ID = "part_H9mK4xR7vT2qN8pL5cY3wBfD"
DEFAULT_PARTNER_NAME = "hermes-agent"
INTEGRATION_REPO = "NousResearch/hermes-agent"
INTEGRATION_VERSION = "1.0.0"
DEFAULT_MODEL = "anthropic/claude-sonnet-5"

MIN_AMOUNT_USD_MINOR = 2_000
MAX_AMOUNT_USD_MINOR = 1_000_000
DEFAULT_AMOUNT_USD_MINOR = 2_500


@dataclass(frozen=True)
class Endpoints:
    auth_base_url: str
    app_base_url: str
    inference_base_url: str
    pay_base_url: str
    verification_base_url: str


def _env_or_default(name: str, fallback: str) -> str:
    return os.getenv(name, "").strip() or fallback


def resolve_endpoints() -> Endpoints:
    return Endpoints(
        auth_base_url=_env_or_default("AIMLAPI_AUTH_URL", "https://auth.aimlapi.com"),
        app_base_url=_env_or_default("AIMLAPI_APP_URL", "https://app.aimlapi.com"),
        inference_base_url=_env_or_default(
            "AIMLAPI_INFERENCE_URL", "https://api.aimlapi.com/v1"
        ),
        pay_base_url=_env_or_default("AIMLAPI_PAY_URL", "https://pay.aimlapi.com"),
        verification_base_url=_env_or_default(
            "AIMLAPI_VERIFICATION_BASE_URL", "https://aimlapi.com/app"
        ),
    )


def resolve_partner_id() -> str:
    return _env_or_default("AIMLAPI_PARTNER_ID", DEFAULT_PARTNER_ID)


def resolve_invite_code() -> str:
    return os.getenv("AIMLAPI_INVITE_CODE", "").strip()


def resolve_return_url(frontend_base_url: str | None = None) -> str:
    return require_trusted_aimlapi_url(
        _env_or_default(
            "AIMLAPI_RETURN_URL",
            frontend_base_url or resolve_endpoints().verification_base_url,
        )
    )


def attribution_headers() -> dict[str, str]:
    return {
        "X-AIMLAPI-Source": "agent",
        "X-AIMLAPI-Partner-ID": resolve_partner_id(),
        "X-AIMLAPI-Integration-Repo": INTEGRATION_REPO,
        "X-AIMLAPI-Integration-Version": INTEGRATION_VERSION,
    }


def is_trusted_aimlapi_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme.lower() == "https"
        and not parsed.username
        and not parsed.password
        and (host == "aimlapi.com" or host.endswith(".aimlapi.com"))
    )


def require_trusted_aimlapi_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not is_trusted_aimlapi_url(normalized):
        raise ValueError("AI/ML API endpoints must use HTTPS on aimlapi.com")
    return normalized


def build_checkout_return_urls(
    session_token: str,
    pay_base_url: str | None = None,
) -> tuple[str, str]:
    base = require_trusted_aimlapi_url(pay_base_url or resolve_endpoints().pay_base_url)
    token = quote(session_token, safe="")
    return (
        f"{base}/checkout?checkout=success&partnerCheckout=1&sessionToken={token}",
        f"{base}/checkout?checkout=cancel&partnerCheckout=1&sessionToken={token}",
    )
