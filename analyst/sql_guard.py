"""Validate LLM-generated SQL: read-only, single statement."""
import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|copy|export|import|"
    r"pragma|install|load|call|set|read_csv\w*|read_parquet|read_json\w*|glob|httpfs)\b", re.I)


class UnsafeSQLError(Exception):
    pass


def validate_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQLError("Empty SQL.")
    no_comments = re.sub(r"--.*?$|/\*.*?\*/", " ", s, flags=re.S | re.M)
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", no_comments)
    if ";" in no_strings:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if not re.match(r"^\s*(select|with)\b", no_strings, re.I):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    m = FORBIDDEN.search(no_strings)
    if m:
        raise UnsafeSQLError(f"Forbidden keyword: {m.group(0)}")
    return s
