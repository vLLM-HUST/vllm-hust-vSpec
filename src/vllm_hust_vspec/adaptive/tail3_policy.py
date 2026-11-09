"""Restrict unchanged read-only6->4 admission to stable slow N1..3 tails."""

from .prefix_shrink import PrefixShrinkPhase


class Tail3PrefixPhase(PrefixShrinkPhase):
    def consider(self):
        parent = self.parent
        memory = getattr(parent, 'cohort_memory', None)
        if (len(parent.running()) > 3 or
                (parent.mode == 'validate' and
                 (memory is None or memory.completed < 2))):
            self.pending = None
            self.host_sample = None
            return
        return super().consider()
