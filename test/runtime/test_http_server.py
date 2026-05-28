"""Tests for the control-plane HTTP server sidecar."""

import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import requests

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 21101
BASE_URL = f"http://{CONTROL_HOST}:{CONTROL_PORT}"
FAKE_GATEWAY = "http://127.0.0.1:29999"


def _start_control_server():
    """Start the control server against a fake gateway URL."""
    from tokenspeed.runtime.entrypoints.http_server import start

    start(gateway_url=FAKE_GATEWAY, host=CONTROL_HOST, port=CONTROL_PORT)


class TestControlHttpServer(unittest.TestCase):
    """Unit-level tests for the control HTTP server endpoints.

    These tests mock aiohttp so no real gateway or engine is needed.
    """

    def _make_mock_resp(self, payload: dict, status: int = 200):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        return mock_resp

    def _make_mock_session(self, payload: dict, status: int = 200):
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=self._make_mock_resp(payload, status))
        session.get = MagicMock(return_value=self._make_mock_resp(payload, status))
        return session

    def test_health_always_ok(self):
        """GET /health returns 200 without hitting the gateway."""
        import threading

        import uvicorn

        from tokenspeed.runtime.entrypoints import http_server as hs

        hs._gateway_url = FAKE_GATEWAY
        config = uvicorn.Config(
            hs.app, host=CONTROL_HOST, port=CONTROL_PORT + 1, log_level="error"
        )
        server = uvicorn.Server(config)

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(1)

        try:
            r = requests.get(
                f"http://{CONTROL_HOST}:{CONTROL_PORT + 1}/health", timeout=5
            )
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "ok")
        finally:
            server.should_exit = True

    def test_readiness_always_ok(self):
        """GET /readiness returns 200 without hitting the gateway."""
        import threading

        import uvicorn

        from tokenspeed.runtime.entrypoints import http_server as hs

        hs._gateway_url = FAKE_GATEWAY
        config = uvicorn.Config(
            hs.app, host=CONTROL_HOST, port=CONTROL_PORT + 2, log_level="error"
        )
        server = uvicorn.Server(config)

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(1)

        try:
            r = requests.get(
                f"http://{CONTROL_HOST}:{CONTROL_PORT + 2}/readiness", timeout=5
            )
            self.assertEqual(r.status_code, 200)
        finally:
            server.should_exit = True


class TestOrchestratorControlPort(unittest.TestCase):
    """Test that --control-port is parsed by split_argv."""

    def test_control_port_parsed(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(
            ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--control-port", "8081"]
        )
        self.assertEqual(result.opts.control_port, 8081)

    def test_control_port_not_in_engine_or_gateway_args(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m", "--control-port", "8081"])
        self.assertNotIn("--control-port", result.engine)
        self.assertNotIn("--control-port", result.gateway)

    def test_no_control_port_defaults_none(self):
        from tokenspeed.cli._argsplit import split_argv

        result = split_argv(["--model", "m"])
        self.assertIsNone(result.opts.control_port)


if __name__ == "__main__":
    unittest.main()
