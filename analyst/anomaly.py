"""Statistical anomaly detection + data-quality checks (no LLM needed)."""
import pandas as pd

COLS = ["table", "row", "column", "value", "expected_range", "score", "reason"]


def detect_anomalies(df: pd.DataFrame, table: str = "", max_rows: int = 50) -> pd.DataFrame:
    """Flag outliers per numeric column: IQR fences confirmed by modified z-score (MAD)."""
    out = []
    for col in df.select_dtypes("number").columns:
        s = df[col].dropna()
        if len(s) < 8 or s.nunique() < 4:
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        med = s.median(); mad = (s - med).abs().median()
        for idx, v in s[(s < lo) | (s > hi)].items():
            mz = 0.6745 * (v - med) / mad if mad else 0.0
            if mad and abs(mz) <= 3.5:
                continue
            side = "above" if v > hi else "below"
            out.append({"table": table, "row": idx, "column": col, "value": v,
                        "expected_range": f"{lo:,.2f} to {hi:,.2f}", "score": round(abs(mz), 2),
                        "reason": f"{v:,.2f} is {side} the normal range (IQR rule) and "
                                  f"{abs(mz):.1f} robust std-devs from the median {med:,.2f}"})
    res = pd.DataFrame(out, columns=COLS)
    return res.sort_values("score", ascending=False).head(max_rows).reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]; issues = []
        if s.isna().mean() > 0.05: issues.append("many missing values")
        if s.nunique(dropna=True) <= 1: issues.append("constant column")
        if pd.api.types.is_string_dtype(s):
            t = s.dropna().astype(str)
            if t.str.strip().ne(t).any(): issues.append("leading/trailing whitespace")
        rows.append({"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                     "missing_pct": round(100 * s.isna().mean(), 1), "unique": int(s.nunique()),
                     "issues": ", ".join(issues) or "ok"})
    rep = pd.DataFrame(rows)
    rep.attrs["duplicate_rows"] = int(df.duplicated().sum())
    return rep
