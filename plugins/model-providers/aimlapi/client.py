"""Small stdlib client for aimlapi.com guided onboarding."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from .config import (
    DEFAULT_PARTNER_NAME,
    Endpoints,
    attribution_headers,
    build_checkout_return_urls,
    is_trusted_aimlapi_url,
    require_trusted_aimlapi_url,
    resolve_invite_code,
    resolve_partner_id,
    resolve_return_url,
)

_MAX_RESPONSE_BYTES = 1_048_576


class APIError(RuntimeError):
    def __init__(self, message: str, *, status: int = 0):
        super().__init__(message)
        self.status = status


class AimlapiClient:
    def __init__(
        self,
        endpoints: Endpoints,
        *,
        timeout: float = 60.0,
        allow_untrusted_test_endpoints: bool = False,
    ):
        self.endpoints = endpoints
        self.timeout = timeout
        self._allow_untrusted_test_endpoints = allow_untrusted_test_endpoints

    def _endpoint(self, base_url: str, path: str) -> str:
        base = base_url.strip().rstrip("/")
        if not self._allow_untrusted_test_endpoints:
            base = require_trusted_aimlapi_url(base)
        return base + path

    def _request(
        self,
        method: str,
        url: str,
        *,
        bearer: str = "",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=payload, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", "hermes-cli")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if bearer.strip():
            request.add_header("Authorization", f"Bearer {bearer.strip()}")
        for name, value in attribution_headers().items():
            request.add_header(name, value)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
            if status in {401, 403} and bearer.strip():
                detail = "authentication was rejected"
            elif len(detail) > 300:
                detail = detail[:300]
            raise APIError(
                f"aimlapi.com request failed ({status})"
                + (f": {detail}" if detail else ""),
                status=status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise APIError(f"aimlapi.com request could not be completed: {exc}") from None

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise APIError("aimlapi.com response exceeded the 1 MB safety limit")
        if status < 200 or status >= 300:
            raise APIError(f"aimlapi.com request failed ({status})", status=status)
        if not raw.strip():
            return {}
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APIError("aimlapi.com returned an invalid JSON response") from None
        if not isinstance(result, dict):
            raise APIError("aimlapi.com returned an unexpected response shape")
        return result

    def check_account(self, email: str) -> str:
        data = self._request(
            "PATCH",
            self._endpoint(self.endpoints.auth_base_url, "/v1/auth/account"),
            body={"email": email},
        )
        action = str(data.get("action") or "")
        if action not in {"sign-in", "sign-up"}:
            raise APIError("aimlapi.com returned an unsupported account action")
        return action

    def send_sign_in_code(self, email: str) -> None:
        self._request(
            "POST",
            self._endpoint(self.endpoints.auth_base_url, "/v1/auth/sign-in/code"),
            body={"email": email},
        )

    def verify_sign_in_code(self, email: str, code: str) -> str:
        data = self._request(
            "POST",
            self._endpoint(
                self.endpoints.auth_base_url,
                "/v1/auth/sign-in/code/verify",
            ),
            body={"email": email, "code": code},
        )
        token = str(data.get("token") or "").strip()
        if not token:
            raise APIError("aimlapi.com did not return an auth token")
        return token

    def create_passwordless_account(self, email: str) -> str:
        body = {"email": email}
        invite_code = resolve_invite_code()
        if invite_code:
            body["inviteCode"] = invite_code
        data = self._request(
            "POST",
            self._endpoint(
                self.endpoints.auth_base_url,
                "/v1/auth/account/passwordless",
            ),
            body=body,
        )
        token = str(data.get("token") or "").strip()
        if not token:
            raise APIError("aimlapi.com did not return an auth token")
        return token

    def create_key(self, bearer: str) -> str:
        data = self._request(
            "POST",
            self._endpoint(self.endpoints.app_base_url, "/v1/keys"),
            bearer=bearer,
            body={"name": "Hermes Agent"},
        )
        key = str(data.get("key") or "").strip()
        if not key:
            raise APIError("aimlapi.com did not return an API key")
        return key

    def create_checkout_session(self) -> str:
        data = self._request(
            "POST",
            self._endpoint(
                self.endpoints.app_base_url,
                "/v3/partner-checkout/sessions",
            ),
            body={
                "partnerId": resolve_partner_id(),
                "partnerName": DEFAULT_PARTNER_NAME,
                "returnUrl": resolve_return_url(self.endpoints.verification_base_url),
            },
        )
        token = str(data.get("sessionToken") or "").strip()
        if not token:
            raise APIError("aimlapi.com did not return a checkout session")
        return token

    def start_checkout(
        self,
        bearer: str,
        session_token: str,
        amount_usd_minor: int,
        payment_session_id: str,
        auto_top_up: bool,
    ) -> str:
        success_url, cancel_url = build_checkout_return_urls(
            session_token,
            self.endpoints.pay_base_url,
        )
        token = urllib.parse.quote(session_token, safe="")
        body: dict[str, Any] = {
            "amountUsdMinor": amount_usd_minor,
            "paymentSessionId": payment_session_id,
            "method": "card",
            "successUrl": success_url,
            "cancelUrl": cancel_url,
        }
        if auto_top_up:
            body["autoTopUp"] = True
        data = self._request(
            "POST",
            self._endpoint(
                self.endpoints.app_base_url,
                f"/v3/partner-checkout/sessions/{token}/pay",
            ),
            bearer=bearer,
            body=body,
        )
        checkout = data.get("checkout")
        pay_url = str(checkout.get("payUrl") if isinstance(checkout, dict) else "")
        return self._validate_checkout_url(pay_url)

    def get_checkout_session(self, session_token: str) -> dict[str, Any]:
        token = urllib.parse.quote(session_token, safe="")
        return self._request(
            "GET",
            self._endpoint(
                self.endpoints.app_base_url,
                f"/v3/partner-checkout/sessions/{token}",
            ),
        )

    def exchange_checkout(self, bearer: str, session_token: str) -> str:
        token = urllib.parse.quote(session_token, safe="")
        data = self._request(
            "POST",
            self._endpoint(
                self.endpoints.app_base_url,
                f"/v3/partner-checkout/sessions/{token}/exchange",
            ),
            bearer=bearer,
        )
        key = str(data.get("apiKey") or "").strip()
        if not key:
            raise APIError("aimlapi.com did not return an API key")
        return key

    @staticmethod
    def new_payment_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _validate_checkout_url(value: str) -> str:
        normalized = value.strip()
        parsed = urllib.parse.urlparse(normalized)
        trusted_checkout = is_trusted_aimlapi_url(normalized) or (
            parsed.scheme.lower() == "https"
            and parsed.hostname == "checkout.stripe.com"
            and not parsed.username
            and not parsed.password
        )
        if not trusted_checkout:
            raise APIError("aimlapi.com returned an invalid checkout URL")
        return normalized

    def wait_for_checkout(
        self,
        session_token: str,
        *,
        timeout: float = 1_500.0,
        poll_interval: float = 2.0,
    ) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data = self.get_checkout_session(session_token)
            except APIError as exc:
                if exc.status and exc.status < 500:
                    raise
                time.sleep(poll_interval)
                continue
            status = str(data.get("status") or "").lower()
            if status == "paid":
                paid_session_token = str(data.get("sessionToken") or "").strip()
                return paid_session_token or session_token
            if status == "exchanged":
                raise APIError(
                    "aimlapi.com checkout was already exchanged; rotate the key "
                    "from the dashboard"
                )
            if status in {"cancelled", "expired", "failed"}:
                raise APIError(f"aimlapi.com checkout ended with status: {status}")
            time.sleep(poll_interval)
        raise APIError("aimlapi.com checkout timed out")
