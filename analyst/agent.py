"""Orchestrator: plan (LLM) -> validate -> execute -> chart -> narrate (LLM, streaming)."""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass
import pandas as pd
from .anomaly import detect_anomalies, quality_report
from .charts import build_chart
from .llm import LLMError, extract_json
from .loader import DataStore
from .sql_guard import UnsafeSQLError, validate_sql

log = logging.getLogger(__name__)
MAX_ROWS = 5000

PLAN_SYSTEM = """You are a senior data analyst. You answer questions about the user's CSV data held in DuckDB.

{schema}

Respond with ONLY a JSON object:
{{
 "intent": "query" | "anomaly" | "quality" | "chitchat",
 "reasoning": "2-4 sentences: how you interpret the question and which tables/columns/logic you use",
 "sql": "a single read-only DuckDB SELECT (or WITH) query; required when intent=query",
 "pandas": "equivalent pandas code as a string (variables named after tables), for display only",
 "chart": {{"type": "bar|line|pie|scatter|histogram|none", "x": "col", "y": "col", "color": "col or null", "title": "..."}},
 "table": "table name (for anomaly/quality intents)",
 "answer": "short reply (only for chitchat)"
}}
Rules: use only existing tables/columns; use LIMIT for large outputs; for monthly trends use date_trunc('month', col);
'underperforming' = lowest/below-average performers on the relevant metric; resolve follow-ups
("now as a pie chart", "same for 2023") using the conversation history; join tables when the question needs it.
Pick a chart only when it helps; the SQL output columns must include the chart's x and y."""

NARRATE_SYSTEM = """You are a business analyst. Using the question, the executed analysis and its result,
write a concise answer (max ~150 words): lead with the direct answer, then 2-3 insights or caveats with
concrete numbers from the result. Never invent numbers. Markdown allowed."""


@dataclass
class Turn:
    question: str
    intent: str = "query"
    reasoning: str = ""
    sql: str = ""
    pandas_code: str = ""
    df: pd.DataFrame | None = None
    fig: object = None
    answer: str = ""
    error: str = ""
    elapsed: float = 0.0


class Analyst:
    def __init__(self, store: DataStore, llm):
        self.store, self.llm = store, llm
        self.history: list[dict] = []

    # ---- step 1: plan + execute ----
    def analyze(self, question: str) -> Turn:
        t0, turn = time.time(), Turn(question)
        if not self.store.frames:
            turn.error = "Please upload at least one CSV file first."
            return turn
        try:
            plan = self._plan(question)
            turn.intent = plan.get("intent", "query")
            turn.reasoning = plan.get("reasoning", "")
            turn.pandas_code = plan.get("pandas", "") or ""
            if turn.intent == "chitchat":
                turn.answer = plan.get("answer", "")
            elif turn.intent == "anomaly":
                self._anomaly(turn, plan)
            elif turn.intent == "quality":
                self._quality(turn, plan)
            else:
                self._query(turn, plan, question)
        except (LLMError, UnsafeSQLError) as e:
            turn.error = str(e)
        except Exception as e:  # never crash the UI
            log.exception("Unexpected failure")
            turn.error = f"Unexpected error: {e}"
        turn.elapsed = round(time.time() - t0, 2)
        log.info("q=%r intent=%s err=%s %.2fs", question, turn.intent, bool(turn.error), turn.elapsed)
        return turn

    def _plan(self, question: str, extra: str = "") -> dict:
        system = PLAN_SYSTEM.format(schema=self.store.schema_text())
        msgs = self.history[-8:] + [{"role": "user", "content": question + extra}]
        return extract_json(self.llm.complete(system, msgs, json_mode=True))

    def _query(self, turn: Turn, plan: dict, question: str) -> None:
        sql = plan.get("sql", "")
        try:
            turn.df = self._run(sql)
        except UnsafeSQLError:
            raise
        except Exception as e:  # one self-correction attempt on SQL errors
            log.warning("SQL failed (%s); retrying with error feedback", e)
            plan = self._plan(question, f"\n\n[Your previous SQL failed.\nSQL: {sql}\nError: {e}\nReturn a corrected JSON plan.]")
            sql = plan.get("sql", "")
            turn.df = self._run(sql)
            turn.reasoning = plan.get("reasoning", turn.reasoning)
            turn.pandas_code = plan.get("pandas", turn.pandas_code) or ""
        turn.sql = sql
        turn.fig = build_chart(turn.df, plan.get("chart"))

    def _run(self, sql: str) -> pd.DataFrame:
        safe = validate_sql(sql)
        return self.store.con.execute(f"SELECT * FROM ({safe}) LIMIT {MAX_ROWS}").df()

    def _targets(self, plan: dict) -> list[str]:
        name = plan.get("table")
        return [name] if name in self.store.frames else list(self.store.frames)

    def _anomaly(self, turn: Turn, plan: dict) -> None:
        res = pd.concat([detect_anomalies(self.store.frames[n], n) for n in self._targets(plan)], ignore_index=True)
        turn.df = res.sort_values("score", ascending=False).reset_index(drop=True)
        turn.pandas_code = turn.pandas_code or ("q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1\n"
                                                "outliers = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]  # + modified z-score > 3.5")
        if not turn.df.empty:
            top = turn.df.head(30)
            top = top.assign(label=top["column"] + " #" + top["row"].astype(str))
            turn.fig = build_chart(top, {"type": "bar", "x": "label", "y": "score",
                                         "title": "Top anomalies (robust z-score)"})

    def _quality(self, turn: Turn, plan: dict) -> None:
        name = self._targets(plan)[0]
        turn.df = quality_report(self.store.frames[name])
        turn.reasoning += f" Table `{name}` has {turn.df.attrs.get('duplicate_rows', 0)} duplicate rows."

    # ---- step 2: streamed narration ----
    def narrate(self, turn: Turn):
        if turn.error or turn.intent == "chitchat":
            yield turn.answer or turn.error
            return
        rows = 0 if turn.df is None else len(turn.df)
        preview = "" if turn.df is None else turn.df.head(25).to_csv(index=False)
        content = (f"Question: {turn.question}\nIntent: {turn.intent}\nReasoning: {turn.reasoning}\n"
                   f"SQL: {turn.sql}\nResult ({rows} rows, first 25 shown):\n{preview}")
        if turn.intent == "anomaly" and rows == 0:
            content += "\nNo anomalies were detected by the IQR/MAD rules."
        yield from self.llm.stream(NARRATE_SYSTEM, [{"role": "user", "content": content}])

    def commit(self, turn: Turn) -> None:
        """Store the turn in memory so follow-up questions have context."""
        a = {"intent": turn.intent, "sql": turn.sql, "summary": turn.answer[:400]}
        self.history += [{"role": "user", "content": turn.question},
                         {"role": "assistant", "content": json.dumps(a)}]
