"""Regenerate the Kenneth French Data Library fixture zips in this directory.

    python tests/fixtures/french/build_fixtures.py

The files reproduce the real library's format byte-for-byte in structure (CRLF line endings,
free-text preamble - including a preamble line that contains commas, a header row with an empty
first field and padded names such as ``"Mom   "``, YYYYMM / YYYYMMDD keys, values in percent,
a blank line + ``Annual Factors: January-December`` section keyed YYYY, missing values coded
-99.99 / -999, and a trailing copyright line). The numbers are illustrative test values, not a
copy of the official series. Zips are written with a fixed timestamp so the output is deterministic.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRLF = "\r\n"
COPYRIGHT = "Copyright 2024 Kenneth R. French"


def _row(key: str, values: list[str]) -> str:
    return key + "," + ",".join(f"{v:>8}" for v in values)


def _fmt(values: list[float | str], decimals: list[int] | None = None) -> list[str]:
    out = []
    for i, v in enumerate(values):
        if isinstance(v, str):
            out.append(v)
        else:
            d = 2 if decimals is None else decimals[i]
            out.append(f"{v:.{d}f}")
    return out


FF3_MONTHLY = [
    ("202301", [6.64, 5.02, -4.05, 0.35]),
    ("202302", [-2.59, 1.21, -0.78, 0.34]),
    ("202303", [2.51, -6.94, -99.99, 0.36]),
    ("202304", [0.61, -2.56, -0.04, 0.35]),
    ("202305", [0.35, -0.38, -7.72, 0.36]),
    ("202306", [6.46, 1.34, -0.26, 0.40]),
    ("202307", [3.21, 2.86, 4.11, 0.45]),
    ("202308", [-2.39, -3.65, -1.06, 0.45]),
    ("202309", [-5.24, -1.80, 1.52, 0.43]),
    ("202310", [-3.18, -4.04, 0.19, 0.47]),
    ("202311", [8.83, -0.12, 1.66, 0.44]),
    ("202312", [4.87, 6.34, 4.93, 0.43]),
    ("202401", [0.70, -5.09, -2.38, 0.47]),
    ("202402", [5.06, -0.24, -3.49, 0.42]),
    ("202403", [2.83, -1.17, 4.19, 0.43]),
    ("202404", [-4.67, -2.42, -0.52, 0.47]),
    ("202405", [4.34, 0.76, -1.66, 0.44]),
    ("202406", [2.77, -3.06, -3.31, 0.41]),
    ("202407", [1.24, 6.80, 5.74, 0.45]),
    ("202408", [1.61, -3.65, -1.13, 0.48]),
]
FF3_ANNUAL = [("2023", [21.69, -3.94, -13.81, 5.00])]

FF3_DAILY = [
    ("20240816", [0.20, -0.32, -0.05, 0.022]),
    ("20240819", [1.01, -0.10, -0.62, 0.022]),
    ("20240820", [-0.24, -0.67, 0.12, 0.022]),
    ("20240821", [0.48, 0.66, -0.16, 0.022]),
    ("20240822", [-0.93, -0.12, 0.85, 0.022]),
    ("20240823", [1.24, 1.97, -0.12, 0.022]),
    ("20240826", [-0.30, 0.21, 0.77, 0.022]),
    ("20240827", [0.15, -0.71, -0.42, 0.022]),
    ("20240828", [-0.65, 0.25, 0.42, 0.022]),
    ("20240829", ["-999", 0.32, 0.30, 0.022]),
    ("20240830", [0.54, -0.11, 0.68, 0.022]),
]

FF5_MONTHLY = [
    ("202301", [6.64, 4.37, -4.10, -2.63, -4.57, 0.35]),
    ("202302", [-2.59, 0.69, -0.80, 0.92, -1.43, 0.34]),
    ("202303", [2.51, -5.55, -8.92, 1.95, -2.38, 0.36]),
    ("202304", [0.61, -3.33, -0.03, 2.39, -99.99, 0.35]),
    ("202305", [0.35, -0.73, -7.75, -1.81, -7.23, 0.36]),
    ("202306", [6.46, 1.52, -0.21, 2.18, -1.62, 0.40]),
    ("202307", [3.21, 2.08, 4.14, -0.57, 0.56, 0.45]),
    ("202308", [-2.39, -3.16, -1.08, 3.43, -2.37, 0.45]),
    ("202309", [-5.24, -1.79, 1.45, 1.86, -0.82, 0.43]),
    ("202310", [-3.18, -3.94, 0.19, 2.47, -0.66, 0.47]),
    ("202311", [8.83, -0.03, 1.64, -3.86, -0.99, 0.44]),
    ("202312", [4.87, 7.32, 4.92, -3.04, 1.32, 0.43]),
    ("202401", [0.70, -5.73, -2.38, 0.68, -0.97, 0.47]),
    ("202402", [5.06, -0.78, -3.49, -1.98, -2.15, 0.42]),
    ("202403", [2.83, -2.48, 4.19, 1.48, 1.18, 0.43]),
    ("202404", [-4.67, -2.39, -0.52, 1.49, -0.30, 0.47]),
    ("202405", [4.34, 0.06, -1.66, 2.98, -0.29, 0.44]),
    ("202406", [2.77, -4.37, -3.31, 0.51, -1.78, 0.41]),
    ("202407", [1.24, 8.28, 5.74, -0.21, 0.43, 0.45]),
    ("202408", [1.61, -3.55, -1.13, 0.85, 0.86, 0.48]),
]
FF5_ANNUAL = [("2023", [21.69, -5.01, -13.79, 2.92, -6.52, 5.00])]

MOM_MONTHLY = [
    ("202302", [-0.06]),
    ("202303", [-2.79]),
    ("202304", [1.56]),
    ("202305", [-99.99]),
    ("202306", [-3.68]),
    ("202307", [-3.25]),
    ("202308", [3.29]),
    ("202309", [0.06]),
    ("202310", [1.99]),
    ("202311", [-6.07]),
    ("202312", [-4.95]),
    ("202401", [5.04]),
    ("202402", [4.88]),
    ("202403", [3.13]),
    ("202404", [-0.40]),
    ("202405", [0.12]),
    ("202406", [4.07]),
    ("202407", [-2.43]),
    ("202408", ["-999"]),
]
MOM_ANNUAL = [("2023", [-14.33])]


def ff3_monthly() -> str:
    lines = [
        "This file was created by CMPT_ME_BEME_RETS using the 202408 CRSP database.",
        "The 1-month TBill return is from Ibbotson and Associates, Inc.",
        "",
        ",Mkt-RF,SMB,HML,RF",
        *[_row(k, _fmt(v)) for k, v in FF3_MONTHLY],
        "",
        " Annual Factors: January-December ",
        ",Mkt-RF,SMB,HML,RF",
        *[_row(k, _fmt(v)) for k, v in FF3_ANNUAL],
        "",
        COPYRIGHT,
        "",
    ]
    return CRLF.join(lines)


def ff3_daily() -> str:
    lines = [
        "This file was created by CMPT_ME_BEME_RETS_DAILY using the 202408 CRSP database.",
        "The Tbill return is the simple daily rate that, over the number of trading days",
        "in the month, compounds to 1-month TBill rate from Ibbotson and Associates Inc.",
        "",
        ",Mkt-RF,SMB,HML,RF",
        *[_row(k, _fmt(v, [2, 2, 2, 3])) for k, v in FF3_DAILY],
        COPYRIGHT,
        "",
    ]
    return CRLF.join(lines)


def ff5_monthly() -> str:
    lines = [
        "This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202408 CRSP database.",
        "The 1-month TBill return is from Ibbotson and Associates Inc.",
        "Missing data are indicated by -99.99 or -999.",
        "",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        *[_row(k, _fmt(v)) for k, v in FF5_MONTHLY],
        "",
        " Annual Factors: January-December ",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        *[_row(k, _fmt(v)) for k, v in FF5_ANNUAL],
        "",
        COPYRIGHT,
        "",
    ]
    return CRLF.join(lines)


def mom_monthly() -> str:
    lines = [
        "This file was created by CMPT_ME_PRIOR_RETS using the 202408 CRSP database.",
        "It contains a momentum factor, constructed from six value-weight portfolios formed using independent "
        "sorts on size and prior return of NYSE, AMEX, and NASDAQ stocks.  Mom is the average of the returns on "
        "two (big and small) high prior return portfolios minus the average of the returns on two low prior "
        "return portfolios.  The portfolios are constructed monthly.  Big means a firm is above the median "
        "market cap on the NYSE at the end of the previous month; small firms are below the median NYSE market "
        "cap.  Prior return is measured from month -12 to - 2.  Firms in the low prior return portfolio are "
        "below the 30th NYSE percentile.  Those in the high portfolio are above the 70th NYSE percentile.",
        "Missing data are indicated by -99.99 or -999.",
        "",
        ",Mom   ",
        *[_row(k, _fmt(v)) for k, v in MOM_MONTHLY],
        "",
        " Annual Factors: January-December ",
        ",Mom   ",
        *[_row(k, _fmt(v)) for k, v in MOM_ANNUAL],
        "",
        COPYRIGHT,
        "",
    ]
    return CRLF.join(lines)


FIXTURES = {
    "F-F_Research_Data_Factors_CSV.zip": ("F-F_Research_Data_Factors.CSV", ff3_monthly),
    "F-F_Research_Data_Factors_daily_CSV.zip": ("F-F_Research_Data_Factors_daily.CSV", ff3_daily),
    "F-F_Research_Data_5_Factors_2x3_CSV.zip": ("F-F_Research_Data_5_Factors_2x3.csv", ff5_monthly),
    "F-F_Momentum_Factor_CSV.zip": ("F-F_Momentum_Factor.CSV", mom_monthly),
}


def build(out_dir: Path = HERE) -> list[Path]:
    written = []
    for zip_name, (member, make) in FIXTURES.items():
        info = zipfile.ZipInfo(member, date_time=(2024, 9, 30, 12, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        path = out_dir / zip_name
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(info, make().encode("ascii"))
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in build():
        print(p)
