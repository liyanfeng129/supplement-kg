"""The human sheet's server: the page, and HTTP in front of the sheet service (spec 0011.04).

A frontend, beside `engine/` (ADR 0002 §1): it owns no workflow logic. It
loads the cases once at start, then turns `GET /api/cases` into the cases,
their saved verdicts and the rulebook, and `POST /api/verdict` into a
`human_sheet.save` call, handing its problems back.

Started from notebook 16 with `start(run_dir)`, which serves on a background
thread and returns at once, or from a shell:

    python web/human_sheet/server.py data/dsld/runs/<run> [--port 8765]

It listens on 127.0.0.1 only; VSCode forwards the port from the SSH box.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.dsld.runs.schema import ClaimVerdict, HumanVerdict, MissedClaim
from engine.repositories.prompts import loader as prompts
from engine.services.extraction_eval import human_sheet

PAGE = Path(__file__).resolve().parent / "index.html"


def start(run_dir: str | Path, port: int = 8765, also: Iterable[str] = ()) -> ThreadingHTTPServer:
    """Load the run's sheet cases and serve the page on `127.0.0.1:port` from a daemon thread.

    `also` is the pilot labels beyond the human part to show, as
    `sheet_cases` takes them. Returns the server; `server.shutdown()` stops
    it. Raises if the port is taken, e.g. by a server an earlier notebook cell
    started and did not stop.
    """
    run_dir = Path(run_dir)
    cases = human_sheet.sheet_cases(run_dir, also)
    by_id = {c.case_id: c for c in cases}
    rulebook = prompts.load_rulebook("extraction_judge")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # quiet: the notebook's output is the page, not a request log

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/cases":
                saved = runs_loader.read_human_verdicts(run_dir)
                self._json(
                    200,
                    {
                        "run": run_dir.name,
                        "cases": [asdict(c) for c in cases],
                        "saved": {k: asdict(v) for k, v in saved.items() if k in by_id},
                        "rulebook": rulebook,
                    },
                )
            else:
                self._json(404, {"error": f"no {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/api/verdict":
                self._json(404, {"error": f"no {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                case = by_id[body["case_id"]]
                verdict = HumanVerdict(
                    case_id=body["case_id"],
                    dsld_id=int(body["dsld_id"]),
                    claims=[ClaimVerdict(**c) for c in body["claims"]],
                    missed=[MissedClaim(**m) for m in body["missed"]],
                    opened_at=body["opened_at"],
                    saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    active_seconds=body["active_seconds"],
                )
            except (ValueError, KeyError, TypeError) as exc:
                self._json(400, {"problems": [f"malformed verdict: {exc!r}"]})
                return
            problems = human_sheet.save(run_dir, case, verdict)
            self._json(200, {"problems": problems, "saved_at": None if problems else verdict.saved_at})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the human sheet for one run's pilot.")
    parser.add_argument("run_dir")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = start(args.run_dir, args.port)
    print(f"human sheet for {Path(args.run_dir).name} on http://127.0.0.1:{args.port}  (Ctrl-C stops)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
