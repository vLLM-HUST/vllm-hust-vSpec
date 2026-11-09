"""Opt-in, loopback-only evaluation endpoint for the pinned vLLM API.

Loaded as a standalone top-level module from the installed evaluation asset
directory, so the native child never imports the vSpec plugin package.
"""

import hmac
import os

from fastapi import APIRouter, HTTPException, Request

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _authorized(request):
    state = request.app.state
    if request.client is None or request.client.host not in LOOPBACK:
        raise HTTPException(status_code=403, detail="Evaluation endpoint is loopback-only")
    supplied = request.headers.get("authorization", "")
    expected = "Bearer " + state.vspec_eval_token
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Evaluation authorization required")
    if state.vspec_eval_engine is None:
        raise HTTPException(status_code=503, detail="Evaluation engine unavailable")
    return state.vspec_eval_engine


class EvaluationEndpointPlugin:
    name = "vspec_eval"
    required_tasks = ("generate",)

    def attach_router(self, app):
        router = APIRouter()

        @router.get("/vspec-evaluation-metadata")
        async def metadata(request: Request):
            client = _authorized(request)
            results = await client.collective_rpc("get_eval_metadata", timeout=120)
            if not isinstance(results, list) or len(results) != 2:
                raise HTTPException(status_code=500, detail="Expected two evaluation workers")
            return {"results": results, "read_only": True}

        @router.get("/vspec-evaluation-operator-proof")
        async def operator_proof(request: Request):
            client = _authorized(request)
            results = await client.collective_rpc("get_eval_operator_proof", timeout=120)
            if not isinstance(results, list) or len(results) != 2:
                raise HTTPException(status_code=500, detail="Expected two operator proof records")
            return {"results": results, "read_only": True}

        @router.post("/vspec-evaluation-reset-caches")
        async def reset_caches(request: Request):
            client = _authorized(request)
            for name in ("reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"):
                if not callable(getattr(client, name, None)):
                    raise HTTPException(status_code=501, detail="Required cache API unavailable")
            ok = await client.reset_prefix_cache(
                reset_running_requests=False, reset_connector=False
            )
            if not ok:
                raise HTTPException(
                    status_code=409, detail="Prefix cache still has active references"
                )
            await client.reset_encoder_cache()
            await client.reset_mm_cache()
            return {
                "prefix_cache_cleared": True,
                "encoder_cache_cleared": True,
                "mm_cache_cleared": True,
                "reset_running_requests": False,
                "reset_connector": False,
                "no_answer_or_kv_preload": True,
            }

        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        if getattr(args, "host", None) not in LOOPBACK or engine_client is None:
            raise RuntimeError("Evaluation endpoint requires a loopback generation server")
        token = os.environ.get("VSPEC_EVAL_TOKEN", "")
        if len(token) < 32:
            raise RuntimeError("Evaluation endpoint requires a private random token")
        state.vspec_eval_engine = engine_client
        state.vspec_eval_token = token
