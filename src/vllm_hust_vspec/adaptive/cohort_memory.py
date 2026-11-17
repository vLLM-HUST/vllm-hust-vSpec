"""Host verified-progress memory, used ONLY at drained request boundaries.

This is a progress proxy, not token-acceptance telemetry or a speed claim.
No labels, question text, generated token contents or device reads are used.
Only an independently validated empty-cohort worker transition may enact the
choice. A live cohort cannot shrink and retains both original growth fences.
"""
from collections import deque
import json

from .verified_progress import REQUESTS, MINIMUM_OUTPUT

WINDOW_STEPS = 7
MINIMUM_WIDE_PROGRESS = 4.5
MINIMUM_PROGRESS_RATIO = 1.10
WINDOWS_REQUIRED = 2


class CohortGammaMemory:
    def __init__(self, *, minimum_completed_cohorts=1):
        if minimum_completed_cohorts not in (1, 2):
            raise ValueError('Coverage may defer reuse by exactly one additional cohort')
        self.minimum_completed_cohorts = minimum_completed_cohorts
        self.completed = 0
        self.preferred = 4
        self.reference_narrow = None
        self.sample = None
        self.samples = {gamma: deque(maxlen=WINDOWS_REQUIRED) for gamma in (4, 6)}
        self.active = False

    def observe(self, step, gamma, request_lengths):
        if type(step) is not int or step < 0 or step % WINDOW_STEPS:
            return
        pairs = tuple(sorted(request_lengths))
        if (gamma not in (4, 6) or len(pairs) != REQUESTS
                or len({key for key, _ in pairs}) != REQUESTS
                or any(type(length) is not int or length < 0 for _, length in pairs)
                or max(length for _, length in pairs) < MINIMUM_OUTPUT):
            self.sample = None
            return
        self.active = True
        roster, total = tuple(key for key, _ in pairs), sum(length for _, length in pairs)
        previous = self.sample
        self.sample = (gamma, roster, step, total)
        if previous is not None and previous[1] != roster:
            for values in self.samples.values():
                values.clear()
            return
        if previous is not None and (step <= previous[2] or total < previous[3]):
            self.samples[gamma].clear()
            return
        if (previous is None or previous[0] != gamma or previous[1] != roster
                or step <= previous[2] or total < previous[3]):
            return
        elapsed = step - previous[2]
        if elapsed < WINDOW_STEPS:
            return
        mean = (total - previous[3]) / (REQUESTS * elapsed)
        if not 0 <= mean <= gamma + 1:
            self.samples[gamma].clear()
            return
        self.samples[gamma].append(mean)

    def finish_cohort(self):
        if not self.active:
            return
        self.completed += 1
        means = {gamma: sum(values) / len(values) for gamma, values in self.samples.items()
                 if len(values) == WINDOWS_REQUIRED}
        if 4 in means:
            self.reference_narrow = means[4]
        wide = means.get(6)
        self.preferred = (6 if wide is not None and self.reference_narrow is not None
                          and self.reference_narrow > 0 and wide >= MINIMUM_WIDE_PROGRESS
                          and wide >= MINIMUM_PROGRESS_RATIO * self.reference_narrow else 4)
        print('[q35-cohort-memory] ' + json.dumps(dict(
            completed_cohorts=self.completed, observed_progress_means=means,
            observed_window_counts={gamma: len(values) for gamma, values in self.samples.items()},
            progress_window_steps=WINDOW_STEPS,
            retained_narrow_progress=self.reference_narrow, next_preferred_gamma=self.preferred,
            no_new_GPU_read=True, no_labels_or_text=True, decision_is_progress_proxy_only=True)), flush=True)
        self.sample, self.active = None, False
        for values in self.samples.values():
            values.clear()

    def boundary_anchor(self, anticipated_batch):
        return (self.preferred if anticipated_batch == REQUESTS
                and self.completed >= self.minimum_completed_cohorts else 4)
