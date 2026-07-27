"""AI/ML API model-provider plugin for Hermes Agent."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from .config import (
    DEFAULT_MODEL,
    attribution_headers,
    is_trusted_aimlapi_url,
    resolve_endpoints,
)
from .onboarding import guided_api_key_setup

logger = logging.getLogger(__name__)

HOTTEST_MODELS = (
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-4.8-fast",
    DEFAULT_MODEL,
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-luna-pro",
    "x-ai/grok-4-5",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.6-flash",
    "alibaba/glm-5.2",
    "alibaba/qwen3.7-max",
    "zhipu/glm-5-2",
    "minimax/minimax-m3",
    "moonshot/kimi-k3",
    "zhipu/glm-5.2",
)

CURATED_MODELS = HOTTEST_MODELS + (
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.4-mini",
    "deepseek/deepseek-v4-flash",
    "xiaomi/mimo-v2.5-pro",
    "tencent/hy3",
)

_CATALOG_MAX_BYTES = 2 * 1024 * 1024
_catalog_cache: dict[str, list[dict[str, Any]]] = {}


def _price_per_token(unit: dict[str, Any]) -> str | None:
    try:
        price = Decimal(str(unit["price"]))
        per = Decimal(str(unit["per"]))
    except (InvalidOperation, KeyError, TypeError):
        return None
    if not price.is_finite() or not per.is_finite() or price < 0 or per <= 0:
        return None
    return str(price / per)


class AimlapiProfile(ProviderProfile):
    def _fetch_catalog(
        self,
        *,
        base_url: str | None,
        timeout: float,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]] | None:
        effective_base = (base_url or self.base_url).strip().rstrip("/")
        if not is_trusted_aimlapi_url(effective_base):
            return None
        if not force_refresh and effective_base in _catalog_cache:
            return _catalog_cache[effective_base]

        request = urllib.request.Request(
            effective_base + "/models?include=capabilities,pricing"
        )
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", "hermes-cli")
        for name, value in attribution_headers().items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(_CATALOG_MAX_BYTES + 1)
                if len(raw) > _CATALOG_MAX_BYTES:
                    return None
                payload = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            logger.debug("fetch_catalog(aimlapi): %s", exc)
            return None

        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return None
        catalog = [entry for entry in entries if isinstance(entry, dict)]
        _catalog_cache[effective_base] = catalog
        return catalog

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        entries = self._fetch_catalog(
            base_url=base_url,
            timeout=timeout,
        )
        if entries is None:
            return None
        hottest = []
        models = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "openai/chat-completions":
                continue
            model_id = str(entry.get("id") or "").strip()
            capabilities = entry.get("capabilities")
            supports_tools = isinstance(capabilities, list) and "tools" in capabilities
            if not model_id or model_id in seen or not supports_tools:
                continue
            info = entry.get("info")
            if isinstance(info, dict) and info.get("isHottest") is True:
                hottest.append(model_id)
            else:
                models.append(model_id)
            if model_id:
                seen.add(model_id)

        ranked_hottest = [
            model_id for model_id in HOTTEST_MODELS if model_id in hottest
        ]
        ranked_hottest.extend(
            model_id for model_id in hottest if model_id not in ranked_hottest
        )
        result = ranked_hottest + models
        return result or None

    def fetch_model_pricing(
        self,
        *,
        base_url: str | None = None,
        timeout: float = 8.0,
        force_refresh: bool = False,
    ) -> dict[str, dict[str, str]]:
        entries = self._fetch_catalog(
            base_url=base_url,
            timeout=timeout,
            force_refresh=force_refresh,
        )
        if entries is None:
            return {}

        result: dict[str, dict[str, str]] = {}
        for entry in entries:
            if entry.get("type") != "openai/chat-completions":
                continue
            model_id = str(entry.get("id") or "").strip()
            pricing = entry.get("pricing")
            units = pricing.get("units") if isinstance(pricing, dict) else None
            if not model_id or not isinstance(units, list):
                continue

            model_prices: dict[str, str] = {}
            for unit in units:
                if (
                    not isinstance(unit, dict)
                    or unit.get("type") != "charge"
                    or unit.get("name") != "token"
                    or unit.get("content") != "text"
                    or unit.get("measure") != "output"
                    or unit.get("phase") != "inference"
                ):
                    continue
                field = {
                    ("provided", "user"): "prompt",
                    ("generated", "model"): "completion",
                    ("cached", "user"): "input_cache_read",
                    ("cache_write", "user"): "input_cache_write",
                }.get((unit.get("origin"), unit.get("author")))
                value = _price_per_token(unit)
                if field and value is not None:
                    model_prices[field] = value
            if "prompt" in model_prices and "completion" in model_prices:
                result[model_id] = model_prices
        return result

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_url = str(context.get("base_url") or self.base_url)
        if not is_trusted_aimlapi_url(base_url):
            return {}, {}
        return {}, {"extra_headers": attribution_headers()}


_endpoints = resolve_endpoints()
aimlapi = AimlapiProfile(
    name="aimlapi",
    aliases=("aiml-api", "ai-ml-api"),
    display_name="aimlapi.com",
    description="aimlapi.com (1000+ models, one-click setup)",
    signup_url="https://aimlapi.com/app/keys",
    env_vars=(
        "AIMLAPI_API_KEY",
        "AIMLAPI_INFERENCE_URL",
        "AIMLAPI_AUTH_URL",
        "AIMLAPI_APP_URL",
        "AIMLAPI_PAY_URL",
        "AIMLAPI_VERIFICATION_BASE_URL",
        "AIMLAPI_RETURN_URL",
    ),
    base_url=_endpoints.inference_base_url,
    models_url=_endpoints.inference_base_url.rstrip("/")
    + "/models?include=capabilities,pricing",
    fallback_models=CURATED_MODELS,
    prefer_live_model_discovery=True,
    default_aux_model=DEFAULT_MODEL,
    supports_vision=True,
    allow_base_url_override=False,
    guided_api_key_setup=guided_api_key_setup,
)

register_provider(aimlapi)
