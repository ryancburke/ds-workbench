"""dtype checks that behave the same on pandas 2 and 3.

pandas 3 stores text in the dedicated `str` dtype, so `s.dtype == object` is False
for ordinary string columns there and `select_dtypes(include="object")` only keeps
matching them through a deprecated fallback.
"""
import pandas as pd


def is_text(s: pd.Series) -> bool:
    """True for string-like columns: object dtype (pandas 2) or str/string dtype (pandas 3)."""
    return pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)


def text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_text(df[c])]
