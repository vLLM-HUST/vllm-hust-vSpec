"""Named post-generation metadata RPC; no tensor values or stream calls."""
class ReadOnlyEvenWorker:
    def get_private_even_width_state(self):
        import os
        r=self.model_runner
        refresh=r._q35_fused_gdn_refresh
        widths=[]
        for gid in (1,2,3):
            b=r.attn_groups[gid][0].get_metadata_builder(0)
            widths.append(sorted(getattr(b,'_vspec_adaptive_spec_state_indices_by_width',{})))
        layers=[m for _,m in r.model.named_modules() if all(hasattr(m,k) for k in ('rearrange_mixed_qkv','A_log','dt_bias','head_k_dim'))]
        assert len(layers)==30
        caches=[dict(conv_shape=list(m.kv_cache[0].shape),conv_dtype=str(m.kv_cache[0].dtype),
                     recurrent_shape=list(m.kv_cache[1].shape),recurrent_dtype=str(m.kv_cache[1].dtype)) for m in layers]
        return dict(pid=os.getpid(),dtype=str(r.dtype),quantization=r.model_config.quantization,
                    configured_capacity=r.speculative_config.num_speculative_tokens,
                    synchronized_gamma=getattr(r,'_vspec_mtp_synchronized_gamma',None),
                    actual_width_buffers_by_group=widths,caches=caches,
                    actual_metadata_frames_by_gamma={str(k):v for k,v in refresh.counts.items()},
                    native_metadata_validation_mode=refresh.mode,
                    Python_metadata_only_not_GPU_replay_counter_proof=True)
