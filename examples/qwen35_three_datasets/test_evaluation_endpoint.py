"""CPU-only safety checks for the opt-in evaluation HTTP surface."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

EVALUATION_DIR = Path(__file__).resolve().parents[2] / "src/vllm_hust_vspec/evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
from evaluation_endpoint import EvaluationEndpointPlugin  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.prefix_ok = True

    async def collective_rpc(self, method, timeout):
        self.calls.append((method, timeout))
        return [{"rank": 0}, {"rank": 1}]

    async def reset_prefix_cache(self, **kwargs):
        self.calls.append(("prefix", kwargs))
        return self.prefix_ok

    async def reset_encoder_cache(self):
        self.calls.append("encoder")

    async def reset_mm_cache(self):
        self.calls.append("mm")


class EndpointTests(unittest.TestCase):
    def _app(self):
        app = FastAPI()
        plugin = EvaluationEndpointPlugin()
        plugin.attach_router(app)
        engine = FakeEngine()
        token = "T" * 48
        with patch.dict(os.environ, {"VSPEC_EVAL_TOKEN": token}):
            asyncio.run(plugin.init_state(engine, app.state, Namespace(host="127.0.0.1")))
        return app, engine, token

    def test_auth_and_loopback_and_two_rank_metadata(self):
        app, engine, token = self._app()
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            self.assertEqual(client.get("/vspec-evaluation-metadata").status_code, 401)
            response = client.get(
                "/vspec-evaluation-metadata", headers={"Authorization": "Bearer " + token}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["results"]), 2)
            self.assertNotIn(token, response.text)
        with TestClient(app, client=("192.0.2.1", 50000)) as client:
            response = client.get(
                "/vspec-evaluation-metadata", headers={"Authorization": "Bearer " + token}
            )
            self.assertEqual(response.status_code, 403)
        self.assertEqual(engine.calls, [("get_eval_metadata", 120)])

    def test_reset_order_and_conflict(self):
        app, engine, token = self._app()
        headers = {"Authorization": "Bearer " + token}
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            engine.prefix_ok = False
            self.assertEqual(
                client.post("/vspec-evaluation-reset-caches", headers=headers).status_code, 409
            )
            self.assertEqual(len(engine.calls), 1)
            engine.prefix_ok = True
            response = client.post("/vspec-evaluation-reset-caches", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["no_answer_or_kv_preload"])
        self.assertEqual(engine.calls[-3][0], "prefix")
        self.assertEqual(engine.calls[-2:], ["encoder", "mm"])

    def test_startup_requires_loopback_and_private_token(self):
        plugin = EvaluationEndpointPlugin()
        app = FastAPI()
        with patch.dict(os.environ, {"VSPEC_EVAL_TOKEN": "T" * 48}):
            with self.assertRaises(RuntimeError):
                asyncio.run(plugin.init_state(FakeEngine(), app.state, Namespace(host="0.0.0.0")))
        with patch.dict(os.environ, {"VSPEC_EVAL_TOKEN": "short"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(plugin.init_state(FakeEngine(), app.state, Namespace(host="127.0.0.1")))


if __name__ == "__main__":
    unittest.main()
