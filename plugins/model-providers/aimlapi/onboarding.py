"""Interactive AI/ML API key acquisition flow for Hermes Agent."""

from __future__ import annotations

import getpass
import math
import os
import re
import sys
import webbrowser

from .client import APIError, AimlapiClient
from .config import (
    DEFAULT_AMOUNT_USD_MINOR,
    MAX_AMOUNT_USD_MINOR,
    MIN_AMOUNT_USD_MINOR,
    resolve_endpoints,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _debug(stage: str, error: Exception) -> None:
    if os.getenv("AIMLAPI_DEBUG", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    print(f"  [AI/ML API debug] {stage}: {error}", file=sys.stderr)


def _retry_idempotent(call):
    try:
        return call()
    except APIError as exc:
        if exc.status and exc.status < 500:
            raise
        return call()


def _prompt_amount() -> int | None:
    default_usd = DEFAULT_AMOUNT_USD_MINOR / 100
    minimum = MIN_AMOUNT_USD_MINOR / 100
    maximum = MAX_AMOUNT_USD_MINOR / 100
    print()
    print(f"  Add credits (min ${minimum:.0f}).")
    while True:
        try:
            raw = input(f"  Amount [{default_usd:.0f}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        try:
            amount = float(raw) if raw else default_usd
        except ValueError:
            print("  Please enter a top-up amount.")
            continue
        if not math.isfinite(amount):
            print("  Please enter a top-up amount.")
            continue
        minor = round(amount * 100)
        if minor < MIN_AMOUNT_USD_MINOR:
            print(f"  Minimum top-up is ${minimum:.0f}.")
            continue
        if minor > MAX_AMOUNT_USD_MINOR:
            print(f"  Maximum top-up is ${maximum:.0f}.")
            continue
        return minor


def _prompt_choice(title: str, labels: list[str], default: int = 0) -> int | None:
    try:
        from hermes_cli.setup import _curses_prompt_choice

        selected = _curses_prompt_choice(title, labels, default)
        return selected if selected is not None and selected >= 0 else None
    except Exception:
        pass

    print()
    print(f"  {title}")
    print("  Select by number, Enter to confirm.")
    print()
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {label}")
    print()
    try:
        raw = input(f"  Choice [default {default + 1}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    try:
        selected = int(raw or str(default + 1)) - 1
    except ValueError:
        return None
    return selected if 0 <= selected < len(labels) else None


def _prompt_auto_top_up() -> bool | None:
    selected = _prompt_choice("Auto top-up.", ["On", "Off"])
    if selected is None:
        return None
    return selected == 0


def _open_and_wait(
    client: AimlapiClient,
    session_token: str,
    checkout_url: str,
) -> None:
    print()
    print("  Opening checkout in browser...")
    print(
        "  If the browser did not open automatically please use this link "
        "to top up your account:"
    )
    print()
    print(f"  {checkout_url}")
    print()
    try:
        webbrowser.open(checkout_url)
    except Exception:
        pass
    print("  Waiting for payment confirmation. Press Ctrl+C to cancel.")
    client.wait_for_checkout(session_token)


def guided_api_key_setup(existing_key: str = "") -> str | None:
    stage = "starting guided setup"
    try:
        if existing_key.strip():
            selected = _prompt_choice(
                "aimlapi.com account is already configured",
                [
                    "Continue with your saved API key",
                    "Set up a new key or switch account",
                ],
            )
            if selected is None:
                return None
            if selected == 0:
                print("Everything is ready.")
                return existing_key.strip()

        selected = _prompt_choice(
            "Do you have aimlapi.com key?",
            ["I am a new user", "I already have aimlapi.com key"],
        )
        if selected is None:
            return None
        if selected == 1:
            print()
            print("  Enter your aimlapi.com key.")
            try:
                api_key = getpass.getpass("  Paste your key: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return None
            if not api_key:
                print(
                    "API key is invalid. Please make sure you enter a valid "
                    "aimlapi.com key."
                )
                return None
            print("Everything is ready.")
            return api_key

        client = AimlapiClient(resolve_endpoints())
        print()
        print("  Enter your email.")
        try:
            email = input("  Email: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        if not _EMAIL_RE.fullmatch(email):
            print("Email format is incorrect.")
            return None

        stage = "checking account"
        action = client.check_account(email)
        if action == "sign-in":
            stage = "sending sign-in code"
            client.send_sign_in_code(email)
            try:
                code = input(f"  Enter the 6-digit code sent to {email}: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return None
            if not re.fullmatch(r"\d{6}", code):
                print("Code is incorrect.")
                return None
            stage = "verifying sign-in code"
            bearer = client.verify_sign_in_code(email, code)
            stage = "creating API key"
            api_key = client.create_key(bearer)
        else:
            stage = "creating passwordless account"
            bearer = client.create_passwordless_account(email)
            amount = _prompt_amount()
            if amount is None:
                return None
            auto_top_up = _prompt_auto_top_up()
            if auto_top_up is None:
                return None
            stage = "creating checkout session"
            session_token = client.create_checkout_session()
            payment_session_id = client.new_payment_session_id()
            stage = "starting checkout"
            checkout_url = _retry_idempotent(
                lambda: client.start_checkout(
                    bearer,
                    session_token,
                    amount,
                    payment_session_id,
                    auto_top_up,
                )
            )
            stage = "waiting for checkout"
            _open_and_wait(client, session_token, checkout_url)
            stage = "exchanging checkout for API key"
            api_key = client.exchange_checkout(bearer, session_token)
            print()
            print(f"Top-up successful - ${amount / 100:.0f} credited to your account")
            print()
            print(f"We've emailed you a magic link to {email}.")
            print("Use it to access your aimlapi.com account and review your usage.")

        print()
        print("Everything is ready.")
        return api_key
    except (APIError, ValueError) as exc:
        _debug(stage, exc)
        print("Sign in failed, please try again.")
        return None
