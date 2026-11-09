"""Host-only conservative trigger using already verified generated lengths.

This is a throughput proxy, NOT per-token draft acceptance telemetry. Two
31-step windows with>=3.5verified output tokens/request/engine step suggest
that trying gamma6 may be worthwhile. The original two fences still decide
whether a proposed growth can actually occur. No labels/text content used.
"""

REQUESTS = 14
WINDOW_STEPS = 31
MINIMUM_OUTPUT = 64
MINIMUM_MEAN_PROGRESS = 3.5
REQUIRED_HIGH_WINDOWS = 2
MAXIMUM_PROGRESS_AT_GAMMA4 = 5.0


class VerifiedProgressTrigger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sample = None
        self.high_windows = 0
        self.last_mean = None

    def observe(self, step, request_lengths):
        pairs = tuple(sorted(request_lengths))
        if (len(pairs) != REQUESTS or len({key for key, _ in pairs}) != REQUESTS
                or not isinstance(step, int) or isinstance(step, bool) or step < 0
                or any(not isinstance(length, int) or isinstance(length, bool) or length < 0
                       for _, length in pairs)):
            self.reset()
            return False
        if max(length for _, length in pairs) < MINIMUM_OUTPUT:
            self.reset()
            return False
        roster, total = tuple(key for key, _ in pairs), sum(length for _, length in pairs)
        previous = self.sample
        if previous is None or previous[0] != roster or step <= previous[1] or total < previous[2]:
            self.sample = roster, step, total
            self.high_windows, self.last_mean = 0, None
            return False
        elapsed = step - previous[1]
        if elapsed < WINDOW_STEPS:
            return False
        self.sample = roster, step, total
        mean = (total - previous[2]) / (REQUESTS * elapsed)
        if mean > MAXIMUM_PROGRESS_AT_GAMMA4:
            self.high_windows, self.last_mean = 0, None
            return False
        self.last_mean = mean
        self.high_windows = self.high_windows + 1 if mean >= MINIMUM_MEAN_PROGRESS else 0
        return self.high_windows >= REQUIRED_HIGH_WINDOWS
