"""Default-off two-fence live q5->q7 growth for the pinned private Q35 lane.

No acceptance, model, tensor math or output-budget change. Cohorts begin at
gamma4. Hot growth uses only periodic verified output suffixes; validate mode
instead forces a coverage trigger. Both queue drains and named GPU fences are
included in generation wall time. A completed bridge is drained again before
binding the actual wide target buffers. Never shrink a live cohort.
"""

from functools import wraps
import json
import os
import threading

from .quiescent_plan import drain_one_existing_frame, growth_readiness, suffix_is_periodic
from .verified_progress import VerifiedProgressTrigger
from vllm_hust_vspec.config import q35_acceptance_growth_enabled, q35_cohort_memory_enabled, q35_tail_prefix_shrink_enabled


class QuiescentPhase:
    def __init__(self, scheduler, mode):
        self.scheduler, self.mode = scheduler, mode
        self.owner_thread = None
        self.transition_id = self.poll_steps = self.drained_frames = 0
        self.pending = self.ready = self.original_roster = None
        self.bridge_wait = self.used = False
        self.verified_progress = (VerifiedProgressTrigger()
                                  if mode in ('on', 'validate') and q35_acceptance_growth_enabled() else None)
        self.cohort_memory = None
        self.tail_prefix = None
        if mode in ('on', 'validate') and q35_tail_prefix_shrink_enabled():
            from .tail3_policy import Tail3PrefixPhase
            self.tail_prefix = Tail3PrefixPhase(self)
        if mode in ('on', 'validate') and q35_cohort_memory_enabled():
            from .cohort_memory import CohortGammaMemory
            self.cohort_memory = CohortGammaMemory(minimum_completed_cohorts=2 if mode == 'validate' else 1)

    def check_thread(self):
        current = threading.get_ident()
        if self.owner_thread is None:
            self.owner_thread = current
        if current != self.owner_thread:
            raise RuntimeError('Private growth requires one engine-core owner thread')

    def running(self):
        return list(self.scheduler.running)

    def roster(self):
        return tuple(sorted(request.request_id for request in self.running()))

    def reset_empty(self):
        if self.tail_prefix is not None:
            self.tail_prefix.reset_empty()
        if self.cohort_memory is not None:
            self.cohort_memory.finish_cohort()
        self.pending = self.ready = self.original_roster = None
        self.bridge_wait = self.used = False
        if self.verified_progress is not None:
            self.verified_progress.reset()

    def consider(self):
        requests = self.running()
        if not requests:
            self.reset_empty()
            return
        self.poll_steps += 1
        if self.cohort_memory is not None and self.poll_steps % 7 == 0:
            self.cohort_memory.observe(self.poll_steps, self.scheduler._vspec_adaptive_controller.current_gamma,
                [(request.request_id, len(request.output_token_ids)) for request in requests])
        #31 is coprime to common checkpoint periods; avoid a32-step polling
        #lattice repeatedly landing in the same rejected boundary phases.
        if self.used or self.pending is not None or self.poll_steps % 31:
            return
        scheduler = self.scheduler
        if (len(requests) != 14 or scheduler._vspec_adaptive_controller.current_gamma != 4
                or scheduler.waiting or scheduler.skipped_waiting
                or scheduler.num_waiting_for_streaming_input
                or any(request.use_structured_output for request in requests)):
            if self.verified_progress is not None:
                self.verified_progress.reset()
            return
        histories = [request.output_token_ids for request in requests]
        if self.mode == 'validate':
            trigger = max(map(len, histories)) >= 640
        else:
            trigger = max(map(len, histories)) >= 64 and sum(suffix_is_periodic(tokens) for tokens in histories) >= 7
        if self.verified_progress is not None:
            high_progress = self.verified_progress.observe(
                self.poll_steps, [(request.request_id, len(request.output_token_ids)) for request in requests])
            if high_progress:
                print('[q35-verified-progress-trigger] ' + json.dumps(dict(
                    engine_steps=self.poll_steps, mean_verified_tokens_per_request_step=self.verified_progress.last_mean,
                    high_windows=self.verified_progress.high_windows, no_new_GPU_read=True)), flush=True)
            trigger = trigger or high_progress
        if trigger:
            self.pending = self.roster()

    def fence(self, core, stage):
        requests = self.running()
        current = self.roster()
        if not requests:
            self.reset_empty()
            return False
        if stage == 'bridge':
            if not growth_readiness(4, 6, len(core.batch_queue),
                                    [request.num_output_placeholders for request in requests],
                                    self.pending, current,
                                    pending_prefill=bool(self.scheduler.waiting or self.scheduler.skipped_waiting),
                                    structured=any(request.use_structured_output for request in requests)):
                self.pending = None
                return False
            self.transition_id += 1
            self.original_roster = current
        else:
            if (core.batch_queue or not set(current).issubset(self.original_roster)
                    or any(request.num_output_placeholders for request in requests)
                    or any(request.use_structured_output for request in requests)):
                raise RuntimeError('Actual-wide stage requires a fully drained original cohort subset')
        receipts = core.model_executor.collective_rpc('fence_private_q35_phase_growth',
                     args=(self.transition_id, stage, self.original_roster))
        if (len(receipts) != 2 or len({row['pid'] for row in receipts}) != 2
                or not all(row['eligible'] for row in receipts)
                or len({row.get('mamba_block_size') for row in receipts}) != 1
                or not all(type(row.get('mamba_block_size')) is int and row['mamba_block_size'] > 0 for row in receipts)):
            print('[q35-phase-rejected] ' + json.dumps(dict(stage=stage, receipts=receipts)), flush=True)
            if stage == 'wide':
                raise RuntimeError('Cannot continue a partially admitted live-growth bridge')
            self.pending = None
            return False
        self.ready = dict(transition_id=self.transition_id, stage=stage, old_gamma=4, new_gamma=6,
                          queue_depth=0, placeholders_zero=True, worker_receipts=receipts,
                          original_roster=list(self.original_roster), current_roster=list(current))
        if stage == 'bridge':
            self.scheduler._vspec_adaptive_controller.force(6, len(requests), 0, reason='quiescent_phase_growth')
            self.pending = None
            self.used = True
        self.bridge_wait = False
        print('[q35-phase-fence] ' + json.dumps(self.ready), flush=True)
        return True


def install_core_step():
    # Called by Scheduler.__init__, while the fully defined EngineCore is
    # already constructing its scheduler and BEFORE it binds self.step_fn.
    # Worker-only plugin imports therefore never eagerly import EngineCore.
    from vllm.v1.engine.core import EngineCore

    original = EngineCore.step_with_batch_queue
    if getattr(original, '_q35_quiescent_growth', False):
        return

    @wraps(original)
    def step(core):
        state = getattr(core.scheduler, '_q35_quiescent_phase', None)
        if state is None:
            return original(core)
        state.check_thread()
        if state.bridge_wait:
            if core.batch_queue:
                state.drained_frames += 1
                return drain_one_existing_frame(core, original)
            state.fence(core, 'wide')
            return original(core)
        tail_prefix = getattr(state, 'tail_prefix', None)
        if tail_prefix is not None:
            tail_prefix.consider()
            if tail_prefix.pending is not None:
                if core.batch_queue:
                    state.drained_frames += 1
                    return drain_one_existing_frame(core, original)
                tail_prefix.fence(core)
                return original(core)
        state.consider()
        if state.pending is not None:
            if core.batch_queue:
                state.drained_frames += 1
                return drain_one_existing_frame(core, original)
            state.fence(core, 'bridge')
        return original(core)

    step._q35_quiescent_growth = True
    EngineCore.step_with_batch_queue = step


def apply_quiescent_growth(settings):
    from vllm_ascend import envs

    mode = envs.VLLM_ASCEND_Q35_QUIESCENT_GROWTH
    if mode not in ('off', 'validate', 'on'):
        raise ValueError('Private live growth is off,validate,on')
    if mode == 'off':
        return False
    if (settings.method != 'mtp' or not settings.adaptive_speculation
            or settings.adaptive_min_gamma != 4 or settings.adaptive_max_gamma != 6
            or not settings.adaptive_async or settings.adaptive_policy != 'online'
            or os.environ.get('HUST_VSPEC_MTP_ASYNC_RUNTIME_SWITCH', '0') == '1'):
        raise RuntimeError('Private live growth requires adaptive MTP4/6')
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init, original_schedule = Scheduler.__init__, Scheduler.schedule
    if getattr(original_init, '_q35_quiescent_growth', False):
        return False

    @wraps(original_init)
    def initialize(scheduler, *args, **kwargs):
        original_init(scheduler, *args, **kwargs)
        config = scheduler.vllm_config
        parallel, model = config.parallel_config, config.model_config
        if (getattr(model.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')
                or str(model.dtype) != 'torch.bfloat16' or getattr(model, 'quantization', None) is not None
                or config.lora_config is not None or parallel.tensor_parallel_size != 2
                or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or not config.scheduler_config.async_scheduling
                or config.speculative_config.method != 'mtp'
                or config.speculative_config.num_speculative_tokens != 6
                or config.max_concurrent_batches != 2):
            raise RuntimeError('Private phase growth requires pinned Q35 BF16 TP2/DP1/PP1 async queue2')
        state = QuiescentPhase(scheduler, mode)
        scheduler._q35_quiescent_phase = state
        controller = scheduler._vspec_adaptive_controller
        original_choose = controller.choose

        def choose(*choose_args, **choose_kwargs):
            if choose_kwargs.get('at_cohort_boundary'):
                if state.cohort_memory is not None:
                    anchor = state.cohort_memory.boundary_anchor(choose_args[0])
                    state.used = anchor == 6
                    print('[q35-cohort-anchor] ' + json.dumps(dict(
                        gamma=anchor, anticipated_batch=choose_args[0],
                        completed_cohorts=state.cohort_memory.completed,
                        empty_scheduler_boundary=True, original_worker_device_fence_retained=True)), flush=True)
                    return controller.force(anchor, choose_args[0], choose_args[1], reason='verified_cohort_memory')
                return controller.force(4, choose_args[0], choose_args[1], reason='quiescent_cohort_anchor4')
            return original_choose(*choose_args, **choose_kwargs)

        controller.choose = choose
        install_core_step()
        print(f'[q35-phase-installed] pid={os.getpid()} mode={mode} live_growth=4to6 two_drained_fences=True', flush=True)

    @wraps(original_schedule)
    def schedule(scheduler, *args, **kwargs):
        state = getattr(scheduler, '_q35_quiescent_phase', None)
        marker = state.ready if state is not None else None
        previous_limit = scheduler.max_num_running_reqs
        if marker is not None:
            # Keep arriving prefills out of the transition's exact roster.
            scheduler.max_num_running_reqs = len(scheduler.running)
        try:
            output = original_schedule(scheduler, *args, **kwargs)
        finally:
            scheduler.max_num_running_reqs = previous_limit
        if marker is not None and output.num_scheduled_tokens:
            actual = set(output.num_scheduled_tokens)
            if output.scheduled_new_reqs or not actual.issubset(marker['original_roster']):
                raise RuntimeError('New requests cannot enter a live-growth transition')
            shrinking = marker['stage'] == 'prefix_shrink'
            expected_width = 4 if marker['stage'] == 'bridge' or shrinking else 6
            next_gamma = 4 if shrinking else 6
            if (output.num_spec_tokens_to_schedule != next_gamma
                    or any(len(output.scheduled_spec_decode_tokens.get(request_id, ())) != expected_width
                           for request_id in actual)):
                raise RuntimeError('Old bridge and first wide proposal widths must remain distinct')
            output.vspec_q35_quiescent_growth_receipt = marker
            state.ready = None
            state.bridge_wait = marker['stage'] == 'bridge'
            print(('[q35-prefix-shrink-scheduled] ' if shrinking else '[q35-phase-scheduled] ') + json.dumps(dict(transition_id=marker['transition_id'],
                  stage=marker['stage'], requests=len(actual), actual_query_width=expected_width + 1,
                  next_draft_gamma=next_gamma, queue_was_drained=True)), flush=True)
        return output

    initialize._q35_quiescent_growth = True
    Scheduler.__init__, Scheduler.schedule = initialize, schedule
    return True
