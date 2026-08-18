import json
import sys

import pytest

from providers import get_provider_profile

aimlapi = get_provider_profile("aimlapi")
assert aimlapi is not None
aimlapi_module = sys.modules["plugins.model_providers.aimlapi"]


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    aimlapi_module._catalog_cache.clear()
    yield
    aimlapi_module._catalog_cache.clear()


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return self._payload


def test_live_catalog_keeps_hottest_chat_and_tool_capable_models(monkeypatch):
    captured = {}
    payload = {
        "data": [
            {
                "id": "anthropic/claude-fable-5",
                "info": {"isHottest": True},
                "type": "openai/chat-completions",
                "capabilities": ["streaming", "tools"],
            },
            {
                "id": "tool-model",
                "info": {"isHottest": False},
                "type": "openai/chat-completions",
                "capabilities": ["streaming", "tools"],
            },
            {
                "id": "chat-only",
                "info": {"isHottest": True},
                "type": "openai/chat-completions",
                "capabilities": ["streaming"],
            },
            {
                "id": "image-model",
                "type": "openai/images/generations",
                "capabilities": [],
            },
        ]
    }

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert aimlapi.fetch_models(api_key="key") == [
        "anthropic/claude-fable-5",
        "chat-only",
        "tool-model",
    ]
    assert captured["request"].get_header("X-aimlapi-partner-id")
    assert (
        captured["request"].get_header("X-aimlapi-source")
        == "agent/hermes-agent"
    )
    assert captured["request"].get_header("User-agent") == "hermes-cli"
    assert captured["request"].get_header("Authorization") is None
    assert "include=capabilities,pricing" in captured["request"].full_url


def test_live_catalog_pricing_is_reused_by_model_picker(monkeypatch):
    calls = []
    payload = {
        "data": [
            {
                "id": "priced-model",
                "type": "openai/chat-completions",
                "capabilities": ["tools"],
                "pricing": {
                    "units": [
                        {
                            "type": "charge",
                            "name": "token",
                            "content": "text",
                            "measure": "output",
                            "origin": "provided",
                            "phase": "inference",
                            "author": "user",
                            "price": 2.5,
                            "per": 1_000_000,
                        },
                        {
                            "type": "charge",
                            "name": "token",
                            "content": "text",
                            "measure": "output",
                            "origin": "generated",
                            "phase": "inference",
                            "author": "model",
                            "price": 10,
                            "per": 1_000_000,
                        },
                        {
                            "type": "charge",
                            "name": "token",
                            "content": "text",
                            "measure": "output",
                            "origin": "cached",
                            "phase": "inference",
                            "author": "user",
                            "price": 0.25,
                            "per": 1_000_000,
                        },
                    ]
                },
            }
        ]
    }

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert aimlapi.fetch_models() == ["priced-model"]
    assert aimlapi.fetch_model_pricing() == {
        "priced-model": {
            "prompt": "0.0000025",
            "completion": "0.00001",
            "input_cache_read": "2.5E-7",
        }
    }
    assert len(calls) == 1


def test_runtime_headers_are_gated_by_endpoint():
    _, trusted = aimlapi.build_api_kwargs_extras(base_url="https://api.aimlapi.com/v1")
    _, untrusted = aimlapi.build_api_kwargs_extras(base_url="https://proxy.example/v1")

    assert trusted["extra_headers"]["X-AIMLAPI-Source"] == "agent/hermes-agent"
    assert trusted["extra_headers"]["X-AIMLAPI-Partner-ID"]
    assert trusted["extra_headers"]["X-AIMLAPI-Integration-Version"] == "1.0.0"
    assert untrusted == {}


def test_plugin_uses_fixed_endpoint_and_guided_setup():
    assert aimlapi.allow_base_url_override is False
    assert aimlapi.prefer_live_model_discovery is True
    assert aimlapi.guided_api_key_setup is not None
    assert aimlapi.display_name == "aimlapi.com"
    assert aimlapi.description == "aimlapi.com (1000+ models, one-click setup)"
    assert aimlapi.default_aux_model == "anthropic/claude-sonnet-5"
    assert aimlapi.fallback_models[:4] == (
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-sol-pro",
    )
