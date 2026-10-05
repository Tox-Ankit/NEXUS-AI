"""
Security and validation guard for analytical SQL queries generated for the chatbot.
Enforces read-only SELECT semantics, limits rows, and blocks injection attacks.
"""

import re
from typing import Tuple

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "UPSERT", "ATTACH", "DETACH",
    "COPY", "EXPORT", "IMPORT", "PRAGMA", "CALL", "EXEC",
    "EXECUTE", "GRANT", "REVOKE", "SET", "INSTALL", "LOAD"
}

FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE
)


def validate_analytical_query(sql: str, max_rows: int = 50) -> Tuple[bool, str, str]:
    """
    Validates that a SQL query is a safe, read-only SELECT statement.
    
    Returns:
        (is_valid, sanitized_sql_or_error_message, reason)
    """
    if not sql or not sql.strip():
        return False, "", "Empty SQL query."

    clean_sql = sql.strip().rstrip(";").strip()

    # Block multiple statements (semicolon chaining)
    if ";" in clean_sql:
        return False, "", "Multiple SQL statements are not permitted."

    # Must start with SELECT or WITH (for CTEs leading to SELECT)
    upper_sql = clean_sql.upper().lstrip()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        return False, "", "Only read-only SELECT queries are permitted."

    # Check for forbidden keywords
    match = FORBIDDEN_PATTERN.search(clean_sql)
    if match:
        return False, "", f"Forbidden SQL keyword detected: '{match.group(1)}'."

    # Enforce LIMIT if not present or replace with max_rows cap
    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", clean_sql, re.IGNORECASE)
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > max_rows:
            clean_sql = re.sub(
                r"\bLIMIT\s+\d+\b",
                f"LIMIT {max_rows}",
                clean_sql,
                flags=re.IGNORECASE
            )
    else:
        clean_sql = f"{clean_sql} LIMIT {max_rows}"

    return True, clean_sql, "Valid read-only analytical query."
