import pytest

from providers import get_provider_profile

assert get_provider_profile("aimlapi") is not None

from plugins.model_providers.aimlapi import onboarding as onboarding_module
from plugins.model_providers.aimlapi.client import APIError
from plugins.model_providers.aimlapi.onboarding import (
    _prompt_amount,
    _retry_idempotent,
    guided_api_key_setup,
)


class _NewAccountClient:
    instances = []

    def __init__(self, _endpoints):
        self.checkout_args = None
        type(self).instances.append(self)

    def check_account(self, email):
        assert email == "user@example.com"
        return "sign-up"

    def create_passwordless_account(self, email):
        assert email == "user@example.com"
        return "auth-token"

    def create_checkout_session(self):
        return "session-token"

    def new_payment_session_id(self):
        return "payment-id"

    def start_checkout(
        self,
        bearer,
        session_token,
        amount_usd_minor,
        payment_session_id,
        auto_top_up,
    ):
        self.checkout_args = (
            bearer,
            session_token,
            amount_usd_minor,
            payment_session_id,
            auto_top_up,
        )
        return "https://staging-pay.aimlapi.com/checkout"

    def wait_for_checkout(self, session_token):
        assert session_token == "session-token"
        return "paid-session-token"

    def exchange_checkout(self, bearer, session_token):
        assert bearer == "auth-token"
        assert session_token == "paid-session-token"
        return "issued-key"


class _ExistingAccountClient:
    def __init__(self, _endpoints):
        pass

    def check_account(self, email):
        assert email == "user@example.com"
        return "sign-in"

    def send_sign_in_code(self, email):
        assert email == "user@example.com"

    def verify_sign_in_code(self, email, code):
        assert email == "user@example.com"
        assert code == "123456"
        return "auth-token"

    def create_key(self, bearer):
        assert bearer == "auth-token"
        return "issued-key"


def test_saved_key_can_be_reused_without_network_or_balance_check(monkeypatch, capsys):
    monkeypatch.setattr(onboarding_module, "_prompt_choice", lambda *_args: 0)
    monkeypatch.setattr(
        onboarding_module,
        "AimlapiClient",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("saved-key flow must not call the API")
        ),
    )

    assert guided_api_key_setup("existing-key") == "existing-key"
    assert "Everything is ready." in capsys.readouterr().out


def test_existing_key_can_be_pasted_without_balance_check(monkeypatch, capsys):
    monkeypatch.setattr(onboarding_module, "_prompt_choice", lambda *_args: 1)
    monkeypatch.setattr(
        onboarding_module.getpass,
        "getpass",
        lambda _prompt: "pasted-key",
    )

    assert guided_api_key_setup("") == "pasted-key"
    assert "Everything is ready." in capsys.readouterr().out


def test_new_account_checkout_matches_canonical_flow(monkeypatch, capsys):
    _NewAccountClient.instances = []
    choices = iter((0, 0))
    monkeypatch.setattr(onboarding_module, "AimlapiClient", _NewAccountClient)
    monkeypatch.setattr(
        onboarding_module,
        "_prompt_choice",
        lambda *_args: next(choices),
    )
    monkeypatch.setattr(onboarding_module.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(
        onboarding_module,
        "color",
        lambda text, code: f"<{code}>{text}</>",
    )
    monkeypatch.setattr(onboarding_module, "_prompt_prefilled", lambda *_args: "25")
    answers = iter(("user@example.com",))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert guided_api_key_setup("") == "issued-key"
    assert _NewAccountClient.instances[0].checkout_args == (
        "auth-token",
        "session-token",
        2500,
        "payment-id",
        True,
    )
    output = capsys.readouterr().out
    assert (
        f"<{onboarding_module.Colors.YELLOW}>  Opening checkout in browser...</>"
        in output
    )
    assert (
        f"<{onboarding_module.Colors.GREEN}>"
        "Top-up successful - $25 credited to your account</>" in output
    )
    assert "We've emailed you a magic link to user@example.com." in output


def test_existing_account_sign_in_uses_six_digit_code(monkeypatch):
    monkeypatch.setattr(
        onboarding_module,
        "AimlapiClient",
        _ExistingAccountClient,
    )
    monkeypatch.setattr(onboarding_module, "_prompt_choice", lambda *_args: 0)
    answers = iter(("user@example.com", "123456"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert guided_api_key_setup("") == "issued-key"


def test_amount_reprompts_for_non_finite_and_below_minimum(monkeypatch, capsys):
    answers = iter(("nan", "10", "25"))
    monkeypatch.setattr(
        onboarding_module,
        "_prompt_prefilled",
        lambda *_args: next(answers),
    )

    assert _prompt_amount() == 2500
    output = capsys.readouterr().out
    assert "Please enter a top-up amount." in output
    assert "Minimum top-up is $20." in output


def test_invalid_email_is_printed_in_red(monkeypatch, capsys):
    monkeypatch.setattr(onboarding_module, "_prompt_choice", lambda *_args: 0)
    monkeypatch.setattr("builtins.input", lambda _prompt: "not-an-email")
    monkeypatch.setattr(
        onboarding_module,
        "color",
        lambda text, code: f"<{code}>{text}</>",
    )

    assert guided_api_key_setup("") is None
    assert (
        f"<{onboarding_module.Colors.RED}>Email format is incorrect.</>"
        in capsys.readouterr().out
    )


def test_idempotent_retry_reuses_same_operation(monkeypatch):
    calls = []
    monkeypatch.setattr(onboarding_module.time, "sleep", lambda _delay: None)

    def operation():
        calls.append("payment-id")
        if len(calls) == 1:
            raise APIError("temporary failure", status=503)
        return "ok"

    assert _retry_idempotent(operation) == "ok"
    assert calls == ["payment-id", "payment-id"]


def test_idempotent_retry_waits_for_billing_user(monkeypatch):
    calls = []
    delays = []
    monkeypatch.setattr(onboarding_module.time, "sleep", delays.append)

    def operation():
        calls.append("payment-id")
        if len(calls) < 5:
            raise APIError("billing user not found", status=404)
        return "ok"

    assert _retry_idempotent(operation) == "ok"
    assert calls == ["payment-id"] * 5
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_idempotent_retry_does_not_retry_other_client_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(onboarding_module.time, "sleep", lambda _delay: None)

    def operation():
        calls.append("payment-id")
        raise APIError("invalid request", status=400)

    with pytest.raises(APIError, match="invalid request"):
        _retry_idempotent(operation)
    assert calls == ["payment-id"]


class _AmbiguousExchangeClient(_NewAccountClient):
    exchange_calls = 0

    def exchange_checkout(self, bearer, session_token):
        type(self).exchange_calls += 1
        raise APIError("response lost", status=0)


def test_one_shot_key_exchange_is_not_retried(monkeypatch, capsys):
    _AmbiguousExchangeClient.exchange_calls = 0
    choices = iter((0, 0))
    monkeypatch.setattr(
        onboarding_module,
        "AimlapiClient",
        _AmbiguousExchangeClient,
    )
    monkeypatch.setattr(
        onboarding_module,
        "_prompt_choice",
        lambda *_args: next(choices),
    )
    monkeypatch.setattr(onboarding_module.webbrowser, "open", lambda _url: True)
    answers = iter(("user@example.com", ""))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert guided_api_key_setup("") is None
    assert _AmbiguousExchangeClient.exchange_calls == 1
    assert "Sign in failed, please try again." in capsys.readouterr().out
