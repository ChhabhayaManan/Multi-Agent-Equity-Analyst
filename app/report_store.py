"""Persist final reports as one JSON file per ticker ."""
import json
import os
from pathlib import Path
from typing import Optional

from templates.schemas.outputs import (CompetitorOutput, DocsOutput,
                                       EventOutput, FundamentalsOutput,
                                       NewsOutput, ReportOutput)
from tools.pinecone_tools import namespace_of

__all__ = ["namespace_of", "save_report", "load_report", "list_reports",
           "SCHEMA_VERSION", "SPECIALIST_CLASSES"]

SCHEMA_VERSION = 2

SPECIALIST_CLASSES = {
    "fundamentals": FundamentalsOutput,
    "competitor": CompetitorOutput,
    "news": NewsOutput,
    "events": EventOutput,
    "docs": DocsOutput,
}


# reports folder, created on first use; REPORTS_DIR overrides the default
def _reports_dir() -> Path:
    d = Path(os.environ.get("REPORTS_DIR", "data/reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# one JSON file per ticker, named by its sanitized namespace
def _path(ticker: str) -> Path:
    return _reports_dir() / f"{namespace_of(ticker)}.json"


# dump report + specialists + run bookkeeping to disk as schema v2
def save_report(ticker: str, company_name: str, report: ReportOutput,
                generated_at: str, specialists: Optional[dict] = None,
                runs: Optional[dict] = None) -> Path:
    specialists = specialists or {}
    dumped = {name: (specialists.get(name).model_dump()
                     if specialists.get(name) is not None else None)
              for name in SPECIALIST_CLASSES}
    payload = {"schema_version": SCHEMA_VERSION,
               "ticker": namespace_of(ticker), "company_name": company_name,
               "generated_at": generated_at, "report": report.model_dump(),
               "specialists": dumped, "runs": runs or {}}
    path = _path(ticker)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


# read one report back into pydantic objects; older files flagged legacy
def load_report(ticker: str) -> Optional[dict]:
    path = _path(ticker)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version", 1) < SCHEMA_VERSION:
        return {"legacy": True,
                "ticker": data.get("ticker", namespace_of(ticker)),
                "company_name": data.get("company_name", ""),
                "generated_at": data.get("generated_at", "")}
    data["report"] = ReportOutput(**data["report"])
    raw = data.get("specialists") or {}
    data["specialists"] = {
        name: (cls(**raw[name]) if raw.get(name) else None)
        for name, cls in SPECIALIST_CLASSES.items()}
    data.setdefault("runs", {})
    return data


# every saved report as a summary row, newest first; bad files skipped
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
