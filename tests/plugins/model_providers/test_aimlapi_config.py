import pytest

from providers import get_provider_profile

assert get_provider_profile("aimlapi") is not None

from plugins.model_providers.aimlapi.config import (
    DEFAULT_PARTNER_ID,
    attribution_headers,
    build_checkout_return_urls,
    is_trusted_aimlapi_url,
    resolve_endpoints,
    resolve_return_url,
)


def test_production_defaults(monkeypatch):
    for name in (
        "AIMLAPI_AUTH_URL",
        "AIMLAPI_APP_URL",
        "AIMLAPI_INFERENCE_URL",
        "AIMLAPI_PAY_URL",
        "AIMLAPI_VERIFICATION_BASE_URL",
        "AIMLAPI_RETURN_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    endpoints = resolve_endpoints()

    assert endpoints.auth_base_url == "https://auth.aimlapi.com"
    assert endpoints.app_base_url == "https://app.aimlapi.com"
    assert endpoints.inference_base_url == "https://api.aimlapi.com/v1"
    assert endpoints.pay_base_url == "https://pay.aimlapi.com"
    assert endpoints.verification_base_url == "https://aimlapi.com/app"
    assert resolve_return_url() == "https://aimlapi.com/app"


def test_staging_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("AIMLAPI_AUTH_URL", "https://auth-staging.aimlapi.com")
    monkeypatch.setenv("AIMLAPI_APP_URL", "https://app-staging.aimlapi.com")
    monkeypatch.setenv("AIMLAPI_INFERENCE_URL", "https://api-staging.aimlapi.com/v1")
    monkeypatch.setenv("AIMLAPI_PAY_URL", "https://staging-pay.aimlapi.com")
    monkeypatch.setenv(
        "AIMLAPI_VERIFICATION_BASE_URL", "https://staging.aimlapi.com/app"
    )

    endpoints = resolve_endpoints()

    assert endpoints.auth_base_url == "https://auth-staging.aimlapi.com"
    assert endpoints.app_base_url == "https://app-staging.aimlapi.com"
    assert endpoints.inference_base_url == "https://api-staging.aimlapi.com/v1"
    assert endpoints.pay_base_url == "https://staging-pay.aimlapi.com"
    assert endpoints.verification_base_url == "https://staging.aimlapi.com/app"
    assert resolve_return_url() == "https://staging.aimlapi.com/app"


def test_explicit_return_url_takes_precedence(monkeypatch):
    monkeypatch.setenv(
        "AIMLAPI_VERIFICATION_BASE_URL",
        "https://staging.aimlapi.com/app",
    )
    monkeypatch.setenv(
        "AIMLAPI_RETURN_URL",
        "https://staging.aimlapi.com/app/complete",
    )

    assert resolve_return_url() == "https://staging.aimlapi.com/app/complete"


def test_explicit_client_environment_is_used_without_env_override(monkeypatch):
    monkeypatch.delenv("AIMLAPI_RETURN_URL", raising=False)

    assert (
        resolve_return_url("https://staging.aimlapi.com/app")
        == "https://staging.aimlapi.com/app"
    )
    success_url, _ = build_checkout_return_urls(
        "session",
        "https://staging-pay.aimlapi.com",
    )
    assert success_url.startswith("https://staging-pay.aimlapi.com/checkout?")


def test_attribution_contract_has_no_tolt_fields(monkeypatch):
    monkeypatch.setenv("AIMLAPI_PARTNER_ID", "part_staging")

    headers = attribution_headers()

    assert headers == {
        "X-AIMLAPI-Source": "agent",
        "X-AIMLAPI-Partner-ID": "part_staging",
        "X-AIMLAPI-Integration-Repo": "NousResearch/hermes-agent",
        "X-AIMLAPI-Integration-Version": "1.0.0",
    }
    assert all("tolt" not in name.lower() for name in headers)

    monkeypatch.delenv("AIMLAPI_PARTNER_ID")
    assert attribution_headers()["X-AIMLAPI-Partner-ID"] == DEFAULT_PARTNER_ID


def test_checkout_return_urls_use_pay_endpoint(monkeypatch):
    monkeypatch.setenv("AIMLAPI_PAY_URL", "https://staging-pay.aimlapi.com")

    success_url, cancel_url = build_checkout_return_urls("session token")

    assert success_url.startswith("https://staging-pay.aimlapi.com/checkout?")
    assert cancel_url.startswith("https://staging-pay.aimlapi.com/checkout?")
    assert "sessionToken=session%20token" in success_url


@pytest.mark.parametrize(
    ("url", "trusted"),
    (
        ("https://api.aimlapi.com/v1", True),
        ("https://api-staging.aimlapi.com/v1", True),
        ("https://staging.aimlapi.com/app", True),
        ("http://api.aimlapi.com/v1", False),
        ("https://aimlapi.com.attacker.test/v1", False),
        ("https://token@api.aimlapi.com/v1", False),
    ),
)
def test_trusted_endpoint_filter(url, trusted):
    assert is_trusted_aimlapi_url(url) is trusted
