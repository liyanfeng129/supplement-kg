"""The reveal's server: the page, and HTTP in front of the reveal service (0011.03 Q15).

A frontend, beside `engine/` (ADR 0002 §1): it owns no workflow logic. It builds
the labels once at start, then turns `GET /api/labels` into the labels, the
decisions so far and the rulebook, and `POST /api/decision` into a `reveal.save`
call, handing its problems back.

Started from notebook 16 with `start(run_dir, cases, judged)`, which serves on a
background thread and returns at once. It listens on 127.0.0.1 only; VSCode
forwards the port from the SSH box.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.prompts import loader as prompts
from engine.services.extraction_eval import reveal
from engine.services.extraction_eval.schema import SheetCase

PAGE = Path(__file__).resolve().parent / "index.html"


def start(run_dir: str | Path, cases: list[SheetCase], judged: dict[str, dict[str, dict]], port: int = 8766) -> ThreadingHTTPServer:
    """Build the reveal and serve the page on `127.0.0.1:port` from a daemon thread.

    `cases` and `judged` are as `reveal_labels` takes them. Returns the server;
    `server.shutdown()` stops it. Raises if the port is taken, e.g. by a server
    an earlier notebook cell started and did not stop.
    """
    run_dir = Path(run_dir)
    labels = reveal.reveal_labels(run_dir, cases, judged)
    by_id = {x.case_id: x for x in labels}
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
            elif self.path == "/api/labels":
                self._json(
                    200,
                    {
                        "run": run_dir.name,
                        "judges": list(judged),
                        "labels": [asdict(x) for x in labels],
                        "decided": [asdict(a) for a in runs_loader.read_adjudications(run_dir).values()],
                        "rulebook": rulebook,
                    },
                )
            else:
                self._json(404, {"error": f"no {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/api/decision":
                self._json(404, {"error": f"no {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                label = by_id[body["case_id"]]
                problems = reveal.save(run_dir, label, body["item"], body["final"], body["gap"])
            except (ValueError, KeyError, TypeError) as exc:
                self._json(400, {"problems": [f"malformed decision: {exc!r}"]})
                return
            self._json(200, {"problems": problems})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
