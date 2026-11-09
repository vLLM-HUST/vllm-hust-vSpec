"""Exact CPU sufficient-count memoization; not KV, tokens or model caching."""
from collections import OrderedDict
from .even_round import EvenRoundSelector

class MemoizedEvenRoundSelector(EvenRoundSelector):
    def __init__(self, **options):
        super().__init__(**options)
        self._window_count_cache=OrderedDict();self._deeper_count_cache=OrderedDict()
        self.window_count_hits=self.window_count_misses=0
        self.deeper_count_hits=self.deeper_count_misses=0

    def _request_totals(self, rows, clip_gamma=None):
        # Counts are immutable tuples in production. Normalize lists too so a
        # caller mutation produces a different content key, never stale data.
        key=tuple((tuple(r[2]),tuple(r[3])) for r in rows)
        entry=self._window_count_cache.get(key)
        if entry is None:
            entry={};self._window_count_cache[key]=entry
        self._window_count_cache.move_to_end(key)
        while len(self._window_count_cache)>4:self._window_count_cache.popitem(last=False)
        if clip_gamma not in entry:
            self.window_count_misses+=1
            entry[clip_gamma]=tuple(EvenRoundSelector._request_totals(rows,clip_gamma))
        else:self.window_count_hits+=1
        # Preserve the original fresh-list interface; consumers cannot mutate
        # a cache entry through a returned object. No floating score is cached.
        return list(entry[clip_gamma])

    def _cached_deeper_counts(self, profile, gamma, higher):
        key=(gamma,higher,tuple((tuple(a),tuple(c)) for a,c in profile))
        cached=self._deeper_count_cache.get(key)
        if cached is None:
            self.deeper_count_misses+=1
            cached=dict(
                observed_total=sum(len(a) for a,c in profile),
                surviving=sum(a>=gamma and c>=gamma+1 for aa,cc in profile for a,c in zip(aa,cc)),
                observed_previous=sum(a>=gamma-1 and c>=gamma for aa,cc in profile for a,c in zip(aa,cc)),
                useful_extra=sum(max(0,min(c,min(a,higher)+1)-(gamma+1))
                    for aa,cc in profile for a,c in zip(aa,cc) if a>=gamma and c>=gamma+1))
            self._deeper_count_cache[key]=cached
        else:self.deeper_count_hits+=1
        self._deeper_count_cache.move_to_end(key)
        while len(self._deeper_count_cache)>32:self._deeper_count_cache.popitem(last=False)
        return cached

    def _measured_higher_tail_bound(self, higher, rows, full):
        context=self.context;gamma=self.current_gamma
        if not (isinstance(context,tuple) and context and type(context[0]) is int):return None
        key=(context[0],higher);profile=self.higher_tail_profiles.get(key,())
        if len(profile)<4:return None
        cached=self._cached_deeper_counts(profile,gamma,higher)
        observed_total=cached['observed_total']
        surviving=cached['surviving']
        current_total=sum(len(r[2]) for r in rows)
        if surviving<4 or not current_total:return None
        current_fraction=full/current_total;observed_fraction=surviving/observed_total
        previous=sum(a>=gamma-1 and c>=gamma for r in rows for a,c in zip(r[2],r[3]))
        observed_previous=cached['observed_previous']
        if previous<4 or observed_previous<4:return None
        current_conditional=full/previous;observed_conditional=surviving/observed_previous
        shift=abs(current_conditional/observed_conditional-1)
        self.higher_tail_checks+=1
        result=dict(method='live_measured_deeper_tail_conditioned_on_surviving_current_prefix',
            batch=context[0],observed_gamma=higher,observed_frames=len(profile),
            observed_surviving_current_prefixes=surviving,
            current_full_prefix_fraction=current_fraction,observed_full_prefix_fraction=observed_fraction,
            shift_basis='conditional_survival_of_last_current_prefix_token',
            current_previous_prefixes=previous,observed_previous_prefixes=observed_previous,
            current_last_prefix_conditional_survival=current_conditional,
            observed_last_prefix_conditional_survival=observed_conditional,
            relative_prefix_distribution_shift=shift,existing_change_trigger=self.change_trigger,
            last_observed_policy_round=self.higher_tail_last_round[key],
            age_in_policy_observations=self.rounds-self.higher_tail_last_round[key],
            no_labels_imported_profile_or_fixed_winner=True)
        if shift>=self.change_trigger:
            self.higher_tail_shifted+=1
            result.update(applied=False,reason='material_live_prefix_distribution_change')
            return result
        useful_extra=cached['useful_extra']
        conditional_extra=useful_extra/surviving
        result.update(applied=True,observed_committed_extra=useful_extra,
            conditional_extra_per_surviving_prefix=conditional_extra,
            extra_bound_for_current_window=full*conditional_extra)
        self.higher_tail_applied+=1
        return result

    def summary(self):
        result=super().summary()
        result['exact_CPU_count_memoization']=dict(
            window_hits=self.window_count_hits,window_misses=self.window_count_misses,
            deeper_hits=self.deeper_count_hits,deeper_misses=self.deeper_count_misses,
            current_window_entries=len(self._window_count_cache),maximum_window_entries=4,
            current_deeper_entries=len(self._deeper_count_cache),maximum_deeper_entries=32,
            keys_are_full_immutable_integer_count_content_not_identity_or_age=True,
            only_integer_sufficient_counts_cached_no_score_latency_or_winner=True,
            fresh_list_return_prevents_alias_mutation=True,
            cache_lifetime_one_engine_never_saved_or_shared=True,
            no_KV_model_tensors_token_ids_answers_or_device_reads=True,
            identical_policy_for_identical_feedback_required=True)
        return result
