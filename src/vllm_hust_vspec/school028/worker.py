"""Compose with the selected worker without editing the installed runtime."""

import inspect
import os
from functools import wraps

from . import event, validate_proposer


def make_worker(base):
    class Worker(base):
        def load_model(self, *args, **kwargs):
            import torch
            from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

            original = AscendSpecDecodeBaseProposer.compute_draft_token_ids
            if original.__module__ != "vllm_ascend.spec_decode.llm_base_proposer":
                raise ValueError("Draft token selector already replaced")
            if getattr(original, "__wrapped__", None) is not None:
                raise ValueError("Draft token selector already wrapped")
            active = os.environ["HUST_VSPEC_SCHOOL_MODE"] == "on"
            seen = set()

            @wraps(original)
            def compute(proposer, hidden_states, sampling_metadata=None):
                validate_proposer(proposer, sampling_metadata)
                if active:
                    from vllm_hust_vspec.backends.mtp import _sample_mtp_local_argmax

                    result = _sample_mtp_local_argmax(proposer, hidden_states)
                else:
                    result = original(proposer, hidden_states, sampling_metadata)
                rows = hidden_states.shape[0]
                if rows not in seen:
                    seen.add(rows)
                    event("selector_executed", rank=self.rank, rows=rows, active=active)
                return result

            AscendSpecDecodeBaseProposer.compute_draft_token_ids = compute
            result = super().load_model(*args, **kwargs)
            proposer = self.model_runner.drafter
            validate_proposer(proposer, None)
            model = proposer.model
            weight = model.lm_head.weight
            if weight.dtype != torch.bfloat16 or weight.shape[1] != 2048:
                raise ValueError("Expected Qwen3.5 BF16 head")
            with torch.inference_mode():
                for rows in (1, 24, 36, 48):
                    for kind in ("zeros", "sin"):
                        x = torch.arange(rows * 2048, device=weight.device, dtype=torch.float32)
                        x = (x / 17).sin().reshape(rows, 2048).to(weight.dtype)
                        if kind == "zeros":
                            x.zero_()
                        expected, probs = original(proposer, x, None)
                        actual = model.get_top_tokens(x)
                        torch.npu.synchronize()
                        exact = probs is None and torch.equal(actual, expected)
                        event("head_equivalence", rank=self.rank, rows=rows, kind=kind, exact=exact)
                        if not exact:
                            raise ValueError("MTP local/full argmax mismatch")
            event(
                "model_loaded",
                rank=self.rank,
                active=active,
                original_source=inspect.getsourcefile(original),
            )
            return result

    return Worker
