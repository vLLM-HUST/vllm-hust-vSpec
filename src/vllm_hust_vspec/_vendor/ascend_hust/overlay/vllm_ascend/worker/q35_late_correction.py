"""Opt-in later placement of the SAME required accepted-count CPU wait.

Only exact-masked target attention plus admitted device-only GDN may precede
the wait. All native metadata fallbacks flush first. Drafting starts only
after the original CPU correction has run exactly once.
"""

import os
from collections import Counter

import numpy as np
import torch

from vllm_ascend.worker.q35_fused_gdn import BUFFER_NAMES, _same_metadata


def uniform_late_frame(drafts, scheduled, decode_drafts, with_prefill):
    if with_prefill or drafts is None or not len(drafts):
        return False
    n = len(drafts)
    if n > 16 or len(scheduled) != n or len(decode_drafts) != n:
        return False
    gamma = int(drafts[0])
    return (
        gamma in (4, 6)
        and np.all(drafts == gamma)
        and np.all(scheduled == gamma + 1)
        and np.all(decode_drafts == gamma)
    )


class Q35LateCorrection:
    def __init__(self, runner, mode, fused_refresh_mode):
        if mode not in ("off", "validate", "on"):
            raise ValueError("Late correction mode must be off, validate, or on")
        self.runner, self.mode = runner, mode
        self.pending_n = None
        self.counts = Counter()
        self.snapshots = None
        self.had_pending = False
        self.enabled = (
            mode != "off"
            and fused_refresh_mode == "on"
            and runner._q35_prepare.enabled
            and runner._q35_masked_fia.enabled
            and runner._q35_masked_fia.mode == ("validate" if mode == "validate" else "on")
            and runner.use_async_spec_decode
            and not runner.use_dcp
            and not runner.use_compress
            and runner.lora_config is None
            and runner.cache_config.mamba_cache_mode == "align"
        )
        if mode != "off":
            print(f"[q35-late-correction-init] pid={os.getpid()} mode={mode} enabled={self.enabled}", flush=True)

    def defer(self, n, drafts, scheduled):
        if not self.enabled:
            return False
        assert self.pending_n is None, "Previous correction was not consumed"
        r = self.runner
        decode = r.num_decode_draft_tokens.cpu[:n].numpy()
        if not uniform_late_frame(drafts, scheduled, decode, r.with_prefill):
            return False
        self.pending_n, self.had_pending = n, True
        self.counts["deferred"] += 1
        if self.mode == "validate":
            self.snapshots = (r.prev_positions.np.copy(), r.prev_num_draft_tokens.np.copy())
        return True

    def flush(self, reason):
        if self.pending_n is None:
            return False
        r, n = self.runner, self.pending_n
        if self.snapshots is not None:
            assert np.array_equal(r.prev_positions.np, self.snapshots[0])
            assert np.array_equal(r.prev_num_draft_tokens.np, self.snapshots[1])
        # This is the original Event.synchronize plus original correction.
        # No count read is moved before it and the event is not replaced.
        r._correct_optimistic_seq_lens_cpu(n)
        if self.mode == "validate":
            torch.testing.assert_close(r.optimistic_seq_lens_cpu[:n], r.seq_lens[:n].cpu(), rtol=0, atol=0)
        self.pending_n, self.snapshots = None, None
        if self.mode != "validate":
            self.had_pending = False
        self.counts[reason] += 1
        if self.counts[reason] in (1, 2, 128, 1024):
            print(f"[q35-late-correction] pid={os.getpid()} reason={reason} counts={dict(self.counts)}", flush=True)
        return True

    def before_metadata(self, device_only_gdn_admitted):
        if not device_only_gdn_admitted:
            self.flush("native_metadata_fallback")

    def after_metadata(self, pending_gdn, fused_results, metadata):
        r = self.runner
        if self.pending_n is not None and self.mode == "validate":
            # Clone GPU results built while CPU bounds were optimistic. Only
            # then wait, run the native builders with corrected CPU lengths,
            # and compare every original full GDN buffer and metadata field.
            builders = [r.attn_groups[gid][0].get_metadata_builder(0) for gid, _ in pending_gdn]
            snapshots = [[getattr(builder, name).clone() for name in BUFFER_NAMES] for builder in builders]
            assert fused_results is not None
            n = self.pending_n
            self.flush("diagnostic_before_target")
            for builder, (_, common), candidate, snapshot in zip(builders, pending_gdn, fused_results, snapshots):
                native = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common,
                    num_accepted_tokens=r.num_accepted_tokens.gpu[:n],
                    num_decode_draft_tokens_cpu=r.num_decode_draft_tokens.cpu[:n],
                )
                _same_metadata(candidate, native)
                for name, expected in zip(BUFFER_NAMES, snapshot):
                    assert torch.equal(getattr(builder, name), expected), f"Late GDN mismatch: {name}"
            gamma = int(r.num_decode_draft_tokens.cpu[0])
            self.counts[f"parity_gamma{gamma}"] += 1
            count = self.counts[f"parity_gamma{gamma}"]
            if count in (1, 2, 128):
                print(f"[q35-late-gdn-parity] pid={os.getpid()} gamma={gamma} checks={count}", flush=True)
        if self.had_pending and self.mode == "validate":
            # The masked backend keeps its conservative rounded bounds. Its
            # diagnostic TND shadow must instead use corrected exact lengths.
            seen = set()
            batches = metadata if isinstance(metadata, list) else [metadata]
            for batch in batches:
                for item in batch.values():
                    if hasattr(item, "seq_lens_list") and id(item) not in seen:
                        seen.add(id(item))
                        count = len(item.seq_lens_list)
                        actual = r.input_batch.num_reqs
                        assert count >= actual
                        item.seq_lens_list = r.optimistic_seq_lens_cpu[:actual].tolist() + [1] * (count - actual)
            self.had_pending = False
