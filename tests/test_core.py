import json
import pandas as pd, pytest
from analyst.agent import Analyst
from analyst.anomaly import detect_anomalies, quality_report
from analyst.llm import detect_provider, extract_json
from analyst.loader import DataStore, DataValidationError, parse_csv, table_name
from analyst.sql_guard import UnsafeSQLError, validate_sql

CSV = open("data/sales.csv", "rb").read()


class FakeLLM:
    def __init__(self, plans): self.plans = list(plans); self.calls = 0
    def complete(self, system, messages, max_tokens=0, **kw):
        self.calls += 1; return json.dumps(self.plans.pop(0))
    def stream(self, system, messages, max_tokens=0): yield from ["Done ", "ok"]


def store():
    s = DataStore(); s.add("sales.csv", CSV); s.add("customers.csv", open("data/customers.csv", "rb").read()); return s


@pytest.mark.parametrize("name,raw", [("a.txt", b"x,y\n1,2"), ("a.csv", b""), ("a.csv", b"x,y\n")])
def test_invalid_files(name, raw):
    with pytest.raises(DataValidationError): parse_csv(name, raw)


def test_loader_cleans_names_and_dates():
    df = parse_csv("x.csv", b"Order Date,Total $\n2024-01-01,5\n2024-02-01,6\n")
    assert list(df.columns) == ["order_date", "total"] and str(df.order_date.dtype).startswith("datetime")
    assert table_name("2024 Sales.csv") == "t_2024_sales"


@pytest.mark.parametrize("sql", ["DROP TABLE sales", "select 1; select 2", "SELECT * FROM read_csv('/etc/passwd')",
                                 "", "delete from sales", "WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a"])
def test_guard_blocks(sql):
    with pytest.raises(UnsafeSQLError): validate_sql(sql)


def test_guard_allows():
    assert validate_sql("SELECT region, SUM(revenue) FROM sales WHERE region='set' GROUP BY 1;")


def test_anomalies_found():
    res = detect_anomalies(pd.read_csv("data/sales.csv"), "sales")
    assert res["value"].max() >= 180 * 600 and res["reason"].str.contains("IQR").all()


def test_quality_missing():
    rep = quality_report(pd.read_csv("data/sales.csv"))
    assert rep.set_index("column").loc["region", "missing"] == 2


def test_agent_query_and_followup_context():
    plan = {"intent": "query", "reasoning": "r", "sql": "SELECT region, SUM(revenue) AS revenue FROM sales GROUP BY 1",
            "chart": {"type": "bar", "x": "region", "y": "revenue"}}
    a = Analyst(store(), FakeLLM([plan, plan]))
    t = a.analyze("revenue by region"); t.answer = "".join(a.narrate(t)); a.commit(t)
    assert len(t.df) == 5  # 4 regions + NULL group and t.fig is not None and t.answer == "Done ok"
    assert len(a.history) == 2


def test_agent_self_corrects_bad_sql():
    bad = {"intent": "query", "sql": "SELECT nope FROM sales"}
    good = {"intent": "query", "sql": "SELECT COUNT(*) AS n FROM sales", "chart": {"type": "none"}}
    llm = FakeLLM([bad, good]); t = Analyst(store(), llm).analyze("count")
    assert not t.error and t.df.iloc[0, 0] == 600 and llm.calls == 2


def test_agent_blocks_unsafe_and_requires_data():
    t = Analyst(store(), FakeLLM([{"intent": "query", "sql": "DROP TABLE sales"}])).analyze("x")
    assert "Only SELECT" in t.error
    assert "upload" in Analyst(DataStore(), FakeLLM([])).analyze("x").error.lower()


def test_agent_anomaly_intent():
    t = Analyst(store(), FakeLLM([{"intent": "anomaly", "table": "sales"}])).analyze("anomalies?")
    assert not t.df.empty and t.fig is not None


def test_provider_detection():
    assert detect_provider("AIzaSyXXXX") == "gemini" and detect_provider("sk-ant-abc") == "anthropic"
    assert detect_provider("") is None and detect_provider("random") is None


def test_extract_json_with_noise():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
