"""Test-wide defaults: the CLI's default data provider is the free real-data one on a user's PC;
tests use the offline synthetic market and never open a browser."""

import pytest


@pytest.fixture(autouse=True)
def _offline_cli_defaults(monkeypatch):
    monkeypatch.setenv("AITRADING_PROVIDER", "synthetic")
    monkeypatch.setenv("AITRADING_NO_BROWSER", "1")
