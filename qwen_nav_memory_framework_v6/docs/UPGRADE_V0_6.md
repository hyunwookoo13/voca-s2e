# Upgrade v0.6: GaP-lite validation sidecar

v0.6 upgrades v0.5 with a minimal set of Graph-as-Policy-inspired engineering features. It does **not** implement a full policy graph interpreter.

## Added

```text
nav_memory_qwen/policy.py
  NAV_SKILL_CARDS
  apply_gap_lite_validation()
  candidate_ref_indexes()
  validation checkpoint helpers
  validation feedback summarizer

Memory context
  schema_version = nav_memory_context_v6
  candidate_refs.exits
  candidate_refs.revisits
  nav_skill_cards
  policy_harness_state

VLM output extension
  selected_candidate_ref for action=go
  memory_ops[*].candidate_ref for revisit/merge operations

Agent
  GaP-lite validation runs after sanitize_vlm_output and before memory_ops commit
  StepResult stores validation_checkpoints and validation_feedback
  save_run() writes Feedback.md
```

## Main safety changes

```text
Unknown go candidate_ref -> request_observation
Missing go candidate_ref -> request_observation
avoid=true candidate_ref -> request_observation
Unknown revisit candidate_ref -> drop memory op
spatial_plausibility.accepted=false -> drop same-place memory op
```

The core principle is unchanged: duplicate nodes are safer than false-positive same-place merges.

## Test additions

```text
test_unknown_go_candidate_ref_is_recovered_with_observation
test_avoid_true_go_candidate_ref_is_blocked
test_valid_go_candidate_ref_passes
test_spatially_implausible_revisit_memory_op_is_dropped
test_revisit_candidate_ref_resolves_to_node_id
test_agent_context_and_feedback_log_include_gap_lite
```

## Explicit non-goals

```text
full Graph-as-Policy interpreter
automatic policy graph generation
online policy graph self-learning
pose graph optimization
```
