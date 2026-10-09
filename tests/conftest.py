"""Test-wide defaults: the CLI's default data provider is the free real-data one on a user's PC;
tests use the offline synthetic market, never open a browser and keep ``AITRADING_HOME`` (saved
strategies, the idea inbox, belief ledgers) in a temporary folder."""

import pytest


@pytest.fixture(autouse=True)
def _offline_cli_defaults(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("AITRADING_PROVIDER", "synthetic")
    monkeypatch.setenv("AITRADING_NO_BROWSER", "1")
    # learning is on by default in the CLI: never write a belief ledger into the developer's home folder
    monkeypatch.setenv("AITRADING_HOME", str(tmp_path_factory.mktemp("aitrading_home")))
