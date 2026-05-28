"""Tests for the direct HTTP server (no smg gateway)."""

import json
import multiprocessing as mp
import os
import sys
import time
import unittest

import requests

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
)

HOST = "127.0.0.1"
PORT = 21100
BASE_URL = f"http://{HOST}:{PORT}"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _launch_server():
    from tokenspeed.runtime.entrypoints.http_server import main

    main(
        [
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--model",
            MODEL,
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.7",
            "--max-model-len",
            "4096",
        ]
    )


def _wait_ready(
    base_url: str, timeout: float = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/readiness", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


class TestHttpServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.proc = mp.Process(target=_launch_server, daemon=True)
        cls.proc.start()
        if not _wait_ready(BASE_URL):
            cls.proc.kill()
            raise RuntimeError("HTTP server failed to start")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.proc.pid)

    def test_health(self):
        r = requests.get(f"{BASE_URL}/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_readiness(self):
        r = requests.get(f"{BASE_URL}/readiness")
        self.assertEqual(r.status_code, 200)

    def test_v1_models(self):
        r = requests.get(f"{BASE_URL}/v1/models")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("data", data)
        self.assertGreater(len(data["data"]), 0)

    def test_v1_completions(self):
        r = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"][0]["text"]) > 0)

    def test_pause_continue_not_available(self):
        """pause/continue return 501 until PR #270 is merged."""
        r = requests.post(f"{BASE_URL}/pause_generation", json={"mode": "abort"})
        # Either 200 (PR #270 merged) or 501 (not yet available)
        self.assertIn(r.status_code, (200, 501))

    def test_flush_cache(self):
        r = requests.post(f"{BASE_URL}/flush_cache")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
