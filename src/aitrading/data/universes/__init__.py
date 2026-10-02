"""Bundled ticker universes for the free-data provider.

``us_starter.csv`` - 150 liquid US-listed common stocks spread across all 11 GICS sectors with a
deliberate tilt to mid-caps (columns ``ticker,name,gics_sector``; sector labels use GICS naming).
It is a convenience starting point for running on a PC, not an index: pass your own tickers or a
universe file for anything else. Membership was chosen in 2026 and is not survivorship-free.

The CSV ships as package data (``[tool.setuptools.package-data] aitrading = ["data/universes/*.csv"]``)
and is read with :mod:`importlib.resources`, so it works from a wheel, a zip or an editable install.
"""

from __future__ import annotations

import csv
import io
from importlib import resources

import pandas as pd

STARTER_FILE = "us_starter.csv"


def read_starter_csv(filename: str = STARTER_FILE) -> str:
    return resources.files(__name__).joinpath(filename).read_text(encoding="utf-8")


def load_starter_universe(filename: str = STARTER_FILE) -> pd.DataFrame:
    """Starter universe indexed by ``ticker`` with columns ``name`` and ``gics_sector``."""
    rows = list(csv.DictReader(io.StringIO(read_starter_csv(filename))))
    df = pd.DataFrame(rows, columns=["ticker", "name", "gics_sector"])
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df.set_index("ticker")


def starter_tickers(filename: str = STARTER_FILE) -> list[str]:
    return list(load_starter_universe(filename).index)
