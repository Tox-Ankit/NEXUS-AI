import os
import re
from typing import List, Optional

import pandas as pd
import polars as pl

SHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")


def extract_spreadsheet_id(url_or_id: str) -> str:
    text = (url_or_id or "").strip()
    match = SHEET_ID_RE.search(text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", text):
        return text
    raise ValueError("Provide a Google Sheets URL or spreadsheet ID.")


class GoogleSheetsLoader:
    @staticmethod
    def credentials_path() -> str:
        return (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()

    @staticmethod
    def _client():
        path = GoogleSheetsLoader.credentials_path()
        if not path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. "
                "Add a service-account JSON path to your .env file."
            )
        if not os.path.isfile(path):
            raise ValueError(f"Google credentials file not found: {path}")
        try:
            import gspread
        except ImportError as exc:
            raise ValueError("Install gspread and google-auth to use Google Sheets.") from exc
        return gspread.service_account(filename=path)

    @staticmethod
    def list_worksheets(url_or_id: str) -> List[str]:
        spreadsheet_id = extract_spreadsheet_id(url_or_id)
        if GoogleSheetsLoader.credentials_path() and os.path.isfile(GoogleSheetsLoader.credentials_path()):
            try:
                client = GoogleSheetsLoader._client()
                book = client.open_by_key(spreadsheet_id)
                return [ws.title for ws in book.worksheets()]
            except Exception:
                pass
        # Fallback for public sheets without service account
        return ["Sheet1"]

    @staticmethod
    def load(url_or_id: str, worksheet: Optional[str] = None) -> pl.DataFrame:
        spreadsheet_id = extract_spreadsheet_id(url_or_id)
        cred_path = GoogleSheetsLoader.credentials_path()
        if cred_path and os.path.isfile(cred_path):
            try:
                client = GoogleSheetsLoader._client()
                book = client.open_by_key(spreadsheet_id)
                ws = book.worksheet(worksheet) if worksheet else book.sheet1
                rows = ws.get_all_records()
                if not rows:
                    values = ws.get_all_values()
                    if not values:
                        raise ValueError("The selected worksheet is empty.")
                    header, *body = values
                    pdf = pd.DataFrame(body, columns=header)
                else:
                    pdf = pd.DataFrame(rows)
                return pl.from_pandas(pdf)
            except Exception as e:
                # If service account errors out, try public export below
                pass

        # Public Google Sheet direct CSV export
        try:
            import requests
            export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
            if worksheet and worksheet != "Sheet1":
                export_url += f"&sheet={worksheet}"
            resp = requests.get(export_url, timeout=15)
            if resp.status_code == 200 and resp.content:
                import io
                return pl.read_csv(io.BytesIO(resp.content), infer_schema_length=10000)
            elif resp.status_code in (401, 403):
                raise ValueError(
                    "This Google Sheet is private. Either share it with 'Anyone with the link can view', "
                    "or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON path in your .env."
                )
            else:
                raise ValueError(f"Google Sheets export returned status code {resp.status_code}")
        except Exception as e:
            if "private" in str(e) or "GOOGLE_APPLICATION_CREDENTIALS" in str(e):
                raise
            raise ValueError(
                f"Failed to load Google Sheet: {str(e)}. "
                "Ensure the sheet is public ('Anyone with the link can view') or configure GOOGLE_APPLICATION_CREDENTIALS."
            )
