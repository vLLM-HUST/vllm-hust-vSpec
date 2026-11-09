"""Dynamic worker extension plus the common evaluation-only RPC."""

from evaluation_worker_common import EvaluationMetadataMixin

from vllm_hust_vspec._vendor.runtime.campaign_candidate_worker import CandidateWorker


class DynamicEvaluationWorker(EvaluationMetadataMixin, CandidateWorker):
    def get_eval_compiler_state(self):
        return self.get_vspec_readonly_compiler_state()

    def get_eval_static_state(self):
        return self.get_private_even_static_state()

    def get_eval_operator_proof(self):
        return self.issue18_campaign_state()
