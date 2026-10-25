"""Export the conversation as a Markdown report."""
from datetime import datetime


def to_markdown(turns) -> str:
    out = [f"# AI Data Analyst Report\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n"]
    for i, t in enumerate(turns, 1):
        out.append(f"## {i}. {t.question}\n")
        if t.reasoning: out.append(f"**Reasoning:** {t.reasoning}\n")
        if t.sql: out.append(f"```sql\n{t.sql}\n```\n")
        if t.df is not None and not t.df.empty:
            out.append("```\n" + t.df.head(20).to_csv(index=False) + "```\n")
        out.append((t.answer or t.error) + "\n")
    return "\n".join(out)
