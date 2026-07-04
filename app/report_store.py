"""Persist final reports as one JSON file per ticker under data/reports/."""
import json
import os
import re
from pathlib import Path
from typing import Optional

from templates.schemas.outputs import ReportOutput


def _reports_dir() -> Path:
    d = Path(os.environ.get("REPORTS_DIR", "data/reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def namespace_of(ticker: str) -> str:
    """HDFCBANK.NS -> HDFCBANK. Strips exchange suffix, keeps alnum, uppercases."""
    base = re.sub(r"\.(NS|BO)$", "", ticker.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9]", "", base).upper()


def _path(ticker: str) -> Path:
    return _reports_dir() / f"{namespace_of(ticker)}.json"


def save_report(ticker: str, company_name: str, report: ReportOutput,
                generated_at: str) -> Path:
    payload = {"ticker": namespace_of(ticker), "company_name": company_name,
               "generated_at": generated_at, "report": report.model_dump()}
    path = _path(ticker)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def load_report(ticker: str) -> Optional[dict]:
    path = _path(ticker)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["report"] = ReportOutput(**data["report"])
    return data


def list_reports() -> list:
    out = []
    for f in _reports_dir().glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"ticker": d["ticker"], "company_name": d["company_name"],
                        "generated_at": d["generated_at"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(out, key=lambda r: r["generated_at"], reverse=True)
