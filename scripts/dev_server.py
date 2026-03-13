#!/usr/bin/env python3

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import socketserver
import threading
import time
from pathlib import Path


WATCH_EXTENSIONS = {".html", ".css", ".js", ".md", ".yml", ".yaml", ".png", ".jpg", ".jpeg", ".svg"}
IGNORED_DIRS = {".git", "__pycache__", "site"}


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in WATCH_EXTENSIONS:
            continue
        stat = path.stat()
        state[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
    return state


class LiveReloadState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.listeners: set[queue.Queue[str]] = set()
        self.current = snapshot(root)

    def add_listener(self) -> queue.Queue[str]:
        listener: queue.Queue[str] = queue.Queue()
        with self.lock:
            self.listeners.add(listener)
        return listener

    def remove_listener(self, listener: queue.Queue[str]) -> None:
        with self.lock:
            self.listeners.discard(listener)

    def notify(self, message: str) -> None:
        with self.lock:
            listeners = list(self.listeners)
        for listener in listeners:
            listener.put(message)

    def watch_forever(self) -> None:
        while True:
            time.sleep(0.75)
            updated = snapshot(self.root)
            if updated != self.current:
                self.current = updated
                self.notify("reload")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, live_reload: LiveReloadState, **kwargs) -> None:
        self.live_reload = live_reload
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/__events":
            self.handle_events()
            return
        if self.path.split("?")[0] == "/slides/topics/_index.json":
            self.handle_topic_index()
            return
        super().do_GET()

    def handle_topic_index(self) -> None:
        topics_dir = Path(self.directory) / "slides" / "topics"
        if not topics_dir.is_dir():
            files: list[str] = []
        else:
            files = sorted(
                p.name for p in topics_dir.iterdir() if p.is_file() and p.suffix == ".md"
            )
        body = json.dumps(files).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def handle_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        listener = self.live_reload.add_listener()
        try:
          self.wfile.write(b"retry: 1000\n\n")
          self.wfile.flush()
          while True:
              try:
                  message = listener.get(timeout=15)
                  self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
              except queue.Empty:
                  self.wfile.write(b": keepalive\n\n")
              self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.live_reload.remove_listener(listener)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reveal.js template with live reload.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1948)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    live_reload = LiveReloadState(root)
    watcher = threading.Thread(target=live_reload.watch_forever, daemon=True)
    watcher.start()

    def handler(*handler_args, **handler_kwargs):
        return Handler(
            *handler_args,
            directory=os.fspath(root),
            live_reload=live_reload,
            **handler_kwargs,
        )

    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {root} at http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")


if __name__ == "__main__":
    main()
