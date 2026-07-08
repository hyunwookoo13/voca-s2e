# Upgrade v0.5 Summary

Added after the spatial aliasing / false-merge review:

1. `MemoryGraph.spatial_plausibility_between_nodes()` computes a graph-composed relative pose between candidate and current/revisit nodes.
2. `verify_revisit_candidate()` now requires VLM confidence, backend visual retrieval support, and spatial plausibility.
3. `verify_merge_request()` also checks spatial plausibility before soft-linking nodes.
4. `get_revisit_candidates()` exposes `spatial_plausibility` to Qwen so the prompt can reason about parallax, baseline, and visual aliasing.
5. `NavAgentConfig` exposes global upper-bound thresholds: `revisit_max_same_place_baseline_m`, `revisit_extended_same_place_baseline_m`, and `revisit_max_same_place_hops`.
6. `PLACE_CATEGORY_SPATIAL_GATE` applies stricter effective gates for corridor, doorway, dead-end, room, intersection, and unknown places. The more conservative endpoint controls the pair.
7. Spatial results now include `same_region_allowed` and `recommended_action`, but same-region does not commit a same-place merge.
8. Prompt instructions now tell Qwen to reject or request more observation when spatial plausibility is false, even if images look similar.
9. Pose graph optimization remains a TODO and is not implemented.

New tests:

```text
test_spatial_plausibility_rejects_far_visual_alias_revisit
test_revisit_candidate_context_exposes_spatial_plausibility
test_corridor_category_gate_rejects_plausible_looking_but_too_far_same_place
```
