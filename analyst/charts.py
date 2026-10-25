"""Build Plotly figures from a chart spec produced by the LLM."""
import pandas as pd
import plotly.express as px


def build_chart(df: pd.DataFrame, spec: dict | None):
    if not spec or spec.get("type") in (None, "none") or df is None or df.empty:
        return None
    t, color, title = spec.get("type"), spec.get("color"), spec.get("title", "")
    cols = set(df.columns)
    x = spec.get("x") if spec.get("x") in cols else df.columns[0]
    nums = df.select_dtypes("number").columns.tolist()
    y = spec.get("y") if spec.get("y") in cols else (nums[0] if nums else df.columns[-1])
    color = color if color in cols else None
    try:
        if t == "bar": return px.bar(df, x=x, y=y, color=color, title=title)
        if t == "line": return px.line(df.sort_values(x), x=x, y=y, color=color, markers=True, title=title)
        if t == "pie": return px.pie(df, names=x, values=y, title=title)
        if t == "scatter": return px.scatter(df, x=x, y=y, color=color, title=title)
        if t == "histogram": return px.histogram(df, x=x, title=title)
    except Exception:
        return None
    return None
