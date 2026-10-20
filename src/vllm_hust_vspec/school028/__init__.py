"""Default-off, source-pinned MTP2 local-argmax experiment."""

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

PINS = {
    "vllm_ascend.spec_decode.llm_base_proposer": (
        "d106cb3cc2e048734fa2986394c43c2177d04190715dc390d8a62d50830cb6e2"
    ),
    "vllm.model_executor.layers.logits_processor": (
        "6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f"
    ),
    "vllm.model_executor.models.interfaces": (
        "e77466ea8c0968f8960aca9570bb73fa9830549827234c5ebaae39a5858a6c23"
    ),
    "vllm.model_executor.models.qwen3_5_mtp": (
        "89b0bde1d11fe5768a634e4365c78274e20694b26f4c6bb68e934328d8d0b190"
    ),
}


def event(kind, **fields):
    path = Path(os.environ["HUST_VSPEC_SCHOOL_EVENT_DIR"])
    with (path / f"vspec-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(event=kind, time=time.time(), **fields)) + "\n")


def validate_proposer(proposer, sampling_metadata):
    if proposer.method != "mtp" or proposer.num_speculative_tokens != 2:
        raise ValueError("School argmax requires native MTP2")
    if sampling_metadata is not None and not sampling_metadata.all_greedy:
        raise ValueError("School argmax requires all-greedy sampling")
    if getattr(proposer, "_enable_probabilistic_draft_probs", False):
        raise ValueError("Probabilistic draft export is not qualified")
    if getattr(proposer.model, "draft_id_to_target_id", None) is not None:
        raise ValueError("Vocabulary remapping is not qualified")


def register():
    if os.environ.get("HUST_VSPEC_SCHOOL028_ARGMAX", "0") != "1":
        return
    if os.environ.get("HUST_VSPEC_ENABLED", "0") != "0":
        raise ValueError("Full vSpec and isolated school adapter are exclusive")
    if os.environ.get("HUST_VSPEC_SCHOOL_MODE") not in ("off", "on"):
        raise ValueError("Explicit OFF/ON arm required")
    if (
        os.environ.get("HUST_MICROFLOW_SCHOOL028_FA") != "1"
        or os.environ.get("HUST_MICROFLOW_ARM") != "on"
    ):
        raise ValueError("This experiment requires the current FA Microflow B1")
    from hust_prefill_microflow.school028 import preflight

    preflight()
    for module, digest in PINS.items():
        spec = importlib.util.find_spec(module)
        if spec is None or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != digest:
            raise ValueError("Unqualified host source: " + module)
    import betterscale.qwen35_worker as host

    from .worker import make_worker

    host.Worker = make_worker(host.Worker)
