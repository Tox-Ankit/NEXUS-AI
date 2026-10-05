import re
from typing import Tuple

_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|EXECUTE|CALL|MERGE|REPLACE|ATTACH|DETACH|VACUUM|PRAGMA|"
    r"INTO\s+OUTFILE|LOAD_FILE|SET\s+ROLE)\b",
    re.IGNORECASE,
)
_SELECT_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_MULTI_STATEMENT = re.compile(r";\s*\S")


def is_read_only_select(sql: str) -> Tuple[bool, str]:
    """Allow only a single SELECT / WITH query. Block writes and multi-statements."""
    if not sql or not str(sql).strip():
        return False, "Query is empty."

    stripped = sql.strip().rstrip(";")
    if _MULTI_STATEMENT.search(sql.strip()):
        return False, "Multiple SQL statements are not allowed."
    if _WRITE_PATTERN.search(stripped):
        return False, "Only read-only SELECT queries are allowed."
    if not _SELECT_START.match(stripped):
        return False, "Query must start with SELECT or WITH."
    return True, "ok"


def enforce_row_limit(sql: str, max_rows: int) -> str:
    """Append LIMIT if the user query does not already include one."""
    stripped = sql.strip().rstrip(";")
    if _LIMIT_PATTERN.search(stripped):
        return stripped
    return f"{stripped} LIMIT {int(max_rows)}"
