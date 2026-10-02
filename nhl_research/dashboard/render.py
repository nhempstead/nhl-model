"""Static HTML research dashboard. No wagering controls."""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nhl_research.config import Paths


def render_dashboard(paths: Paths, payload: dict) -> Path:
    paths.reports.mkdir(parents=True, exist_ok=True)
    json_path = paths.reports / "latest.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    origin = payload.get("origin", "UNKNOWN")
    banner = ""
    if origin == "SYNTHETIC":
        banner = (
            "<div class='banner'>SYNTHETIC OFFLINE FIXTURE — software demonstration only. "
            "These numbers are not historical NHL forecasting or paper-trading results.</div>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>NHL research dashboard</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; color: #111; background: #f7f5f1; }}
    h1,h2 {{ font-weight: 600; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 14px; text-align: left; }}
    .banner {{ background: #5c1010; color: white; padding: 12px; margin-bottom: 16px; }}
    .note {{ color: #444; font-size: 14px; }}
    code {{ background: #eee; padding: 1px 4px; }}
  </style>
</head>
<body>
  {banner}
  <h1>NHL forecasting research dashboard</h1>
  <p class="note">Paper-trading research only. No real-money execution. Origin: <code>{origin}</code></p>
  <h2>Contract</h2>
  <pre>{json.dumps(payload.get("contract", {}), indent=2)}</pre>
  <h2>Evaluation metrics</h2>
  <pre>{json.dumps(payload.get("metrics", {}), indent=2)}</pre>
  <h2>Paired market comparison</h2>
  <pre>{json.dumps(payload.get("paired_vs_market", {}), indent=2)}</pre>
  <h2>Power analysis</h2>
  <pre>{json.dumps(payload.get("power", {}), indent=2)}</pre>
  <h2>Paper ledger summary</h2>
  <pre>{json.dumps(payload.get("ledger", {}), indent=2)}</pre>
  <h2>Forecast sample</h2>
  {_table(payload.get("forecasts_sample", []))}
  <h2>Notes</h2>
  <ul>{''.join(f'<li>{n}</li>' for n in payload.get('notes', []))}</ul>
  <p class="note">Machine-readable copy: <a href="latest.json">latest.json</a></p>
</body>
</html>"""
    html_path = paths.reports / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='note'>No forecast rows.</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def serve(paths: Paths, port: int = 8000) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(paths.reports), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving {paths.reports} at http://127.0.0.1:{port}/")
    server.serve_forever()
