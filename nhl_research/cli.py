"""Command-line interface for the NHL research platform."""

from __future__ import annotations

import argparse
import json
import sys

from nhl_research.config import Paths, load_config
from nhl_research.data.inventory import inventory_payload
from nhl_research.dashboard.render import serve
from nhl_research.pipeline import run_real_smoke, run_synthetic_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NHL point-in-time forecasting research platform (paper trading only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", help="Print verified data-access inventory")
    sub.add_parser("demo", help="Run the labeled synthetic end-to-end demonstration")
    smoke = sub.add_parser("smoke", help="Run chronological evaluation on repository historical files")
    smoke.add_argument("--max-games", type=int, default=0, help="If >0, keep this many most recent games only")
    sub.add_parser("dashboard", help="Serve reports/ as a local dashboard")
    dash = sub.add_parser("serve", help="Alias for dashboard")
    dash.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    paths = Paths.from_config()
    if args.command == "inventory":
        print(json.dumps(inventory_payload(), indent=2))
        return 0
    if args.command == "demo":
        payload = run_synthetic_demo(paths)
        print(json.dumps({k: payload[k] for k in ("origin", "metrics", "ledger", "promotion") if k in payload}, indent=2, default=str))
        print(f"Dashboard written to {paths.reports / 'index.html'}")
        return 0
    if args.command == "smoke":
        payload = run_real_smoke(paths, max_games=args.max_games)
        print(json.dumps(payload.get("metrics", payload), indent=2, default=str))
        return 0
    if args.command in {"dashboard", "serve"}:
        port = getattr(args, "port", 8000)
        serve(paths, port=port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
