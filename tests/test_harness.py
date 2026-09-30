import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import post_generate
from evaluator import evaluate
from report import percentile, write_report


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["stream"] is True
        chunks = [
            {"response": "", "done": False},
            {"response": "4", "done": False},
            {"response": "2", "done": False},
            {"response": "", "done": True, "total_duration": 200000000,
             "load_duration": 10000000, "prompt_eval_count": 7,
             "eval_count": 2, "eval_duration": 100000000},
        ]
        body = "".join(json.dumps(x) + "\n" for x in chunks).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HarnessTests(unittest.TestCase):
    def test_stream_metrics_and_strict_scoring(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = post_generate(f"http://127.0.0.1:{server.server_port}",
                                   {"model": "fake", "prompt": "x", "stream": True}, 5)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["response"], "42")
        self.assertIsNotNone(result["first_text_s"])
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["generation_s"], 0.1)
        self.assertEqual(evaluate("42", {"type": "exact", "value": "42"})["status"], "pass")
        self.assertEqual(evaluate("The answer is 42", {"type": "exact", "value": "42"})["status"], "fail")
        self.assertEqual(evaluate('{"a":1}', {"type": "json", "value": {"a": 1}})["status"], "pass")

    def test_report_and_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 95), 5)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "runs.jsonl"
            first = {
                "run_id": "run-one", "prompt_id": "p1", "category": "arithmetic",
                "model": "fake", "hardware": {"device_label": "test fixture"},
                "phase": "warm", "first_text_s": 0.1, "wall_latency_s": 0.2,
                "tokens_per_s": 20, "load_s": 0.01,
                "quality": {"status": "pass"},
            }
            second = dict(first, run_id="run-two", prompt_id="p2",
                          quality={"status": "fail"})
            source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            csv_path, html_path = write_report(source, Path(tmp) / "report")
            self.assertIn("run-one", csv_path.read_text())
            self.assertIn("run-two", csv_path.read_text())
            self.assertIn("100.0", csv_path.read_text())
            self.assertIn("0.0", csv_path.read_text())
            self.assertIn("test fixture", html_path.read_text())
            self.assertIn("Strict pass rate", html_path.read_text())


if __name__ == "__main__":
    unittest.main()
