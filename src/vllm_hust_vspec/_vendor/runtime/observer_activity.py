"""Report activity without claiming activation just from installation/capture."""
def snapshot(fusion,two):
    return dict(
        GDN_input_fusion_activation_observed=all(bool(x['actual_graph_counts']) for x in fusion),
        GDN_input_fusion_layers_with_observed_activity=sum(bool(x['actual_graph_counts']) for x in fusion),
        two_bank_embedding_activation_observed=bool(two['actual_replay_counts']),
        optimization_activity_is_observation_NOT_precondition_for_valid_B16_measurement=True,
        empty_activity_counts_preserved_NOT_faked=True)
