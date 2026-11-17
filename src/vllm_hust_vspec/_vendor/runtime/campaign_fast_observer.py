"""Post-timer metadata only. No new tensor values, reset or stream calls."""
from .vspec_even_readonly_worker import ReadOnlyEvenWorker
from .observer_activity import snapshot as activity_snapshot
class ReadOnlyEvenFastWorker(ReadOnlyEvenWorker):
    def get_private_even_fast_state(self):
        import torch
        import torch_npu
        result=self.get_private_even_width_state();r=self.model_runner
        records=r._private_frozen_router_records
        assert len(records)==41 and all(record.calls>0 for record in records.values())
        for record in records.values():record.get()
        result['native_router_weight_cache']=dict(receipts=r._private_frozen_router_receipts,
            Python_calls_NOT_device_replay_counts=[record.calls for record in records.values()],
            no_runtime_router_or_shared_gate_approximation=True,
            no_event_removal=True)
        lookup=r._q35_native_lookup_manager
        assert lookup.mode=='on' and len(lookup.helpers)==11 and len(lookup.target_order)==10
        result['native_lookup']=[helper.metrics() for helper in lookup.helpers.values()]
        result['native_lookup_source_unchanged_and_frame_scoped_NOT_cross_step_value_cache']=True
        conv=r._q35_cached_conv_manager.layers
        assert len(conv)==30
        result['native_stride_convolution']=[dict(layer=name,
            **owner._q35_native_conv_tile_helper.metrics()) for name,owner in conv.items()]
        result['native_stride_convolution_no_router_or_event_change']=True
        host=r._q35_native_fia_host_batch
        assert host.mode=='on' and host.enabled
        result['max8_native_attention_host_boxing']=host.metrics()
        result['max8_native_attention_host_boxing_no_twin_graph_no_event_removal']=True
        rgdr=r._q35_rgdr_reuse_manager
        assert rgdr.mode=='on' and len(rgdr.layers)==30
        result['RGDR_state_pipeline']=[dict(layer=name,**helper.metrics()) for name,helper in rgdr.layers.items()]
        result['RGDR_capture_plus_matching_GDN_replay_NOT_independent_RGDR_counter_or_native_shadow']=True
        head=r.drafter._vspec_active_proposal_head
        target=r.model.language_model.lm_head.weight
        assert type(head).__name__=='BalancedCompactProposalHead'
        assert head.weight.dtype==target.dtype==torch.bfloat16
        assert tuple(head.weight.shape)==(49152,2048) and head.ids.numel()==49152
        assert head.weight.untyped_storage().data_ptr()!=target.untyped_storage().data_ptr()
        result['balanced96_proposal_only']=dict(private_shape=list(head.weight.shape),
            BF16=True,shared_target_head_shape=list(target.shape),
            private_weight_does_not_alias_target=True,load_time_independent_full_row_reference_required=True)
        receipt=head._q35_inplace_head_receipt
        assert receipt['mode']=='on' and receipt['complete_original_BF16_weight_bits_exact']
        assert receipt['shared_target_weight_unchanged'] and receipt['original_project_unchanged_ON']
        assert torch_npu.get_npu_format(head.weight)==29
        assert target._version==receipt['shared_target_weight_version']
        fusion=r._q35_gdn_input_manager.metrics()
        assert len(fusion)==30 and all(x['mode']=='on' and x['enabled'] for x in fusion)
        result['private_draft_head_NZ']=receipt
        result['target_GDN_input_fusion_layers']=fusion
        result['GDN_ON_counters_NOT_native_shadow_parity_proof']=True
        replica=fusion[0]['replicated_draft_inputs']
        assert replica['variant']=='embedding' and not replica['validate']
        two=replica['two_bank_embedding']
        assert two['full_replica_tensor_removed']
        result.update(activity_snapshot(fusion,two))
        assert two['original_local_parameter_pointer_and_version_unchanged']
        assert two['target_embedding_storage_and_methods_unchanged']
        result['private_two_bank_embedding']=replica
        layouts=[]
        for role,model in (('target',r.model),('draft',r.drafter.model)):
            for name,layer in model.named_modules():
                receipt=getattr(layer,'_q35_moe_nz_receipt',None)
                if receipt is None:continue
                assert all(w['complete_BF16_weight_bits_exact'] and torch_npu.get_npu_format(getattr(layer,w['name']))==29 for w in receipt['weights'])
                layouts.append(dict(role=role,layer=name,receipt=receipt))
        assert sum(x['role']=='target' for x in layouts)==40 and sum(x['role']=='draft' for x in layouts)==1
        result.update(actual41_NZ_layouts=layouts,
                      compact_body_host_live_update_counts=getattr(r.drafter,'_vspec_compact_body_live_counts',{}),
                      compact_body_shapes=[list(x) for x in sorted(getattr(r.drafter,'_vspec_compact_body_shapes',()))],
                      target_greedy_validation_call_counts={str(k):v for k,v in getattr(r,'_vspec_target_parity_counts',{}).items()},
                      target_greedy_native_checks_at_call_numbers=[1,2,128],
                      metadata_only_NOT_full_model_native_admission=True)
        return result
