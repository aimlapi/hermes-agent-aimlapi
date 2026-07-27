from types import SimpleNamespace

from hermes_cli.model_setup_flows import _prompt_guided_api_key_setup


def _profile(callback):
    return SimpleNamespace(guided_api_key_setup=callback)


def _pconfig(*env_vars):
    return SimpleNamespace(api_key_env_vars=env_vars or ("EXAMPLE_API_KEY",))


def test_guided_setup_saves_returned_key():
    saved = {}

    key, abort = _prompt_guided_api_key_setup(
        _profile(lambda _existing: "issued-key"),
        _pconfig(),
        "",
        save_api_key=saved.__setitem__,
    )

    assert (key, abort) == ("issued-key", False)
    assert saved == {"EXAMPLE_API_KEY": "issued-key"}


def test_guided_setup_passes_existing_key_to_provider():
    received = []

    key, abort = _prompt_guided_api_key_setup(
        _profile(lambda existing: received.append(existing) or "rotated-key"),
        _pconfig(),
        "existing-key",
        save_api_key=lambda *_args: None,
    )

    assert received == ["existing-key"]
    assert (key, abort) == ("rotated-key", False)


def test_guided_setup_cancel_does_not_persist():
    saved = []

    key, abort = _prompt_guided_api_key_setup(
        _profile(lambda _existing: None),
        _pconfig(),
        "existing-key",
        save_api_key=lambda *args: saved.append(args),
    )

    assert (key, abort) == ("existing-key", True)
    assert saved == []


def test_guided_setup_failure_does_not_persist(capsys):
    saved = []

    def fail(_existing):
        raise RuntimeError("provider unavailable")

    key, abort = _prompt_guided_api_key_setup(
        _profile(fail),
        _pconfig(),
        "",
        save_api_key=lambda *args: saved.append(args),
    )

    assert (key, abort) == ("", True)
    assert saved == []
    assert "Guided setup failed." in capsys.readouterr().out


def test_guided_setup_requires_provider_key_env():
    saved = []

    key, abort = _prompt_guided_api_key_setup(
        _profile(lambda _existing: "issued-key"),
        SimpleNamespace(api_key_env_vars=()),
        "",
        save_api_key=lambda *args: saved.append(args),
    )

    assert (key, abort) == ("", True)
    assert saved == []
