from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import CANONICAL_PROVIDERS
from hermes_cli.model_setup_flows import _model_flow_api_key_provider
from providers import get_provider_profile

aimlapi = get_provider_profile("aimlapi")
assert aimlapi is not None


def test_hermes_flow_uses_guided_key_fixed_url_and_live_catalog(monkeypatch):
    assert PROVIDER_REGISTRY["aimlapi"].api_key_env_vars == ("AIMLAPI_API_KEY",)
    assert any(entry.slug == "aimlapi" for entry in CANONICAL_PROVIDERS)
    assert CANONICAL_PROVIDERS[0].slug == "aimlapi"
    assert (
        CANONICAL_PROVIDERS[0].tui_desc == "aimlapi.com (1000+ models, one-click setup)"
    )
    config = {"model": {"default": "old-model"}}
    saved = {}
    picker = {}

    monkeypatch.setattr(
        "hermes_cli.model_setup_flows._prompt_guided_api_key_setup",
        lambda *_args, **_kwargs: ("issued-key", False),
    )
    monkeypatch.setattr(
        aimlapi,
        "fetch_models",
        lambda **_kwargs: ["live-model", "anthropic/claude-sonnet-5"],
    )
    monkeypatch.setattr(
        aimlapi,
        "fetch_model_pricing",
        lambda **_kwargs: {
            "live-model": {
                "prompt": "0.000001",
                "completion": "0.000002",
            }
        },
    )

    def select_model(models, **kwargs):
        picker["models"] = models
        picker["pricing"] = kwargs["pricing"]
        return models[0]

    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        select_model,
    )
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda _model: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _name: "")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.config.save_config",
        lambda value: saved.setdefault("config", value.copy()),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("fixed provider must not prompt for a base URL")
        ),
    )

    _model_flow_api_key_provider(config, "aimlapi", "old-model")

    assert picker["models"] == ["live-model", "anthropic/claude-sonnet-5"]
    assert picker["pricing"]["live-model"]["prompt"] == "0.000001"
    assert saved["config"]["model"]["provider"] == "aimlapi"
    assert saved["config"]["model"]["base_url"] == "https://api.aimlapi.com/v1"


def test_hermes_flow_falls_back_when_live_discovery_raises(monkeypatch):
    assert "aimlapi" in PROVIDER_REGISTRY
    config = {"model": {"default": "old-model"}}
    picker = {}

    monkeypatch.setattr(
        "hermes_cli.model_setup_flows._prompt_guided_api_key_setup",
        lambda *_args, **_kwargs: ("issued-key", False),
    )
    monkeypatch.setattr(
        aimlapi,
        "fetch_models",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    def select_model(models, **_kwargs):
        picker["models"] = models
        return models[0]

    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        select_model,
    )
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda _model: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _name: "")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda _value: None)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("fixed provider must not prompt for a base URL")
        ),
    )

    _model_flow_api_key_provider(config, "aimlapi", "old-model")

    assert picker["models"] == list(aimlapi.fallback_models)
