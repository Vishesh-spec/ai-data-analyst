"""Streamlit UI for the AI Data Analyst."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from analyst.agent import Analyst
from analyst.llm import LLMError, make_llm
from analyst.loader import DataStore, DataValidationError
from analyst.logging_conf import setup_logging
from analyst.report import to_markdown

setup_logging()
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

SAMPLES = ["Which region generated the highest revenue?", "Show monthly sales trends.",
           "Which products are underperforming?", "What are the top five customers?",
           "Generate SQL for this analysis.", "Detect anomalies in the dataset.",
           "Run a data quality check."]

if "store" not in st.session_state:
    st.session_state.update(store=DataStore(), turns=[], agent=None, loaded=set())


def get_agent():
    key = st.session_state.get("api_key") or None
    sig = (key, st.session_state.get("provider"), st.session_state.get("model"))
    if st.session_state.agent is None or st.session_state.get("sig") != sig:
        old = st.session_state.agent
        llm = make_llm(st.session_state.get("provider", "auto"), key, st.session_state.get("model") or None)
        st.session_state.agent = Analyst(st.session_state.store, llm)
        if old: st.session_state.agent.history = old.history
        st.session_state.sig = sig
    return st.session_state.agent


def render(turn, key):
    if turn.error:
        st.error(turn.error)
    if turn.reasoning:
        with st.expander("🧠 Reasoning"):
            st.write(turn.reasoning)
    if turn.fig is not None:
        st.plotly_chart(turn.fig, use_container_width=True, key=f"fig{key}")
    if turn.df is not None and not turn.df.empty:
        st.dataframe(turn.df, use_container_width=True, hide_index=True)
    if turn.sql or turn.pandas_code:
        with st.expander("💻 SQL / Pandas code"):
            if turn.sql: st.code(turn.sql, language="sql")
            if turn.pandas_code: st.code(turn.pandas_code, language="python")


with st.sidebar:
    st.header("📁 Data")
    files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    for f in files or []:
        if (f.name, f.size) in st.session_state.loaded:
            continue
        try:
            name = st.session_state.store.add(f.name, f.getvalue())
            st.success(f"Loaded `{name}`")
        except DataValidationError as e:
            st.error(str(e))
        st.session_state.loaded.add((f.name, f.size))
    for n, df in st.session_state.store.frames.items():
        with st.expander(f"{n} · {len(df):,} rows"):
            st.dataframe(df.head(5), hide_index=True)
    st.header("⚙️ Settings")
    st.selectbox("Provider", ["auto", "gemini", "anthropic"], key="provider",
                 help="auto = detected from your key (AIza… = Gemini, sk-ant-… = Claude)")
    st.text_input("API key", type="password", key="api_key",
                  help="Or set GEMINI_API_KEY / ANTHROPIC_API_KEY in .env")
    st.text_input("Model (optional)", key="model", placeholder="gemini-2.5-flash / claude-sonnet-4-6")
    if st.session_state.turns:
        st.download_button("⬇️ Export report (.md)", to_markdown(st.session_state.turns), "report.md")
    if st.button("🗑️ Reset session"):
        st.session_state.clear(); st.rerun()

st.title("📊 AI Data Analyst")
st.caption("Upload CSVs, then ask questions in plain English. Supports charts, SQL, anomalies and follow-ups.")

for i, t in enumerate(st.session_state.turns):
    with st.chat_message("user"): st.write(t.question)
    with st.chat_message("assistant"):
        st.markdown(t.answer); render(t, i)

if not st.session_state.turns and st.session_state.store.frames:
    st.write("**Try:**")
    cols = st.columns(len(SAMPLES))
    for c, q in zip(cols, SAMPLES):
        if c.button(q, use_container_width=True): st.session_state.pending = q

q = st.chat_input("Ask about your data…") or st.session_state.pop("pending", None)
if q:
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"):
        try:
            agent = get_agent()
            with st.spinner("Analyzing…"):
                turn = agent.analyze(q)
            if not turn.error:
                try:
                    turn.answer = st.write_stream(agent.narrate(turn))
                except LLMError as e:
                    turn.error = str(e)
            render(turn, len(st.session_state.turns))
            if not turn.error: agent.commit(turn)
            st.session_state.turns.append(turn)
        except LLMError as e:
            st.error(f"{e} — check your API key / provider in the sidebar.")
