import os
from typing import List, Optional
from urllib.parse import quote_plus

import pandas as pd
import polars as pl
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.event import listen

from app.security.sql_guard import enforce_row_limit, is_read_only_select

DEFAULT_MAX_ROWS = 50_000
HARD_MAX_ROWS = 200_000


def _apply_read_only(dbapi_conn, _record) -> None:
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    except Exception:
        pass
    finally:
        cursor.close()


class SQLConnector:
    @staticmethod
    def url_from_env(
        host: Optional[str] = None,
        port: Optional[str] = None,
        name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dialect: Optional[str] = None,
    ) -> str:
        dialect = (dialect or os.getenv("DB_DIALECT") or "postgresql").strip().lower()
        if dialect in ("sqlite", "sqlite3"):
            path = name or os.getenv("DB_NAME") or ":memory:"
            return f"sqlite:///{path}"

        host = host if host is not None else os.getenv("DB_HOST", "localhost")
        port = port if port is not None else os.getenv("DB_PORT", "5432")
        name = name if name is not None else os.getenv("DB_NAME", "postgres")
        user = user if user is not None else os.getenv("DB_USER", "postgres")
        password = password if password is not None else os.getenv("DB_PASSWORD", "")
        user_q = quote_plus(user or "")
        pass_q = quote_plus(password or "")
        return f"postgresql+psycopg2://{user_q}:{pass_q}@{host}:{port}/{name}"

    @staticmethod
    def connect(url: str) -> Engine:
        engine = create_engine(url, pool_pre_ping=True)
        listen(engine, "connect", _apply_read_only)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine

    @staticmethod
    def list_tables(engine: Engine) -> List[str]:
        inspector = inspect(engine)
        tables: List[str] = []
        for schema in inspector.get_schema_names():
            if schema in ("pg_catalog", "information_schema"):
                continue
            for table in inspector.get_table_names(schema=schema):
                if schema in ("public", "main", None):
                    tables.append(table)
                else:
                    tables.append(f"{schema}.{table}")
        return sorted(tables)

    @staticmethod
    def list_columns(engine: Engine, table: str) -> List[str]:
        inspector = inspect(engine)
        schema = None
        name = table
        if "." in table:
            schema, name = table.split(".", 1)
        return [col["name"] for col in inspector.get_columns(name, schema=schema)]

    @staticmethod
    def fetch_table(engine: Engine, table: str, max_rows: int = DEFAULT_MAX_ROWS) -> pl.DataFrame:
        ident = _quote_ident(table)
        sql = f"SELECT * FROM {ident}"
        return SQLConnector.fetch_query(engine, sql, max_rows=max_rows)

    @staticmethod
    def fetch_query(
        engine: Engine,
        sql: str,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> pl.DataFrame:
        allowed, reason = is_read_only_select(sql)
        if not allowed:
            raise ValueError(reason)
        cap = min(int(max_rows), HARD_MAX_ROWS)
        safe_sql = enforce_row_limit(sql, cap)
        with engine.connect() as conn:
            pdf = pd.read_sql_query(text(safe_sql), conn)
        return pl.from_pandas(pdf)


def _quote_ident(table: str) -> str:
    if "." in table:
        schema, name = table.split(".", 1)
        return f'"{schema}"."{name}"'
    return f'"{table}"'
