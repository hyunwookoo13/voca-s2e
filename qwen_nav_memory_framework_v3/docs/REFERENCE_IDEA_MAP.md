# Reference Idea Map v0.3

This document explains which research ideas informed the implementation. The package does not vendor external research code; it implements lightweight interfaces that can later be replaced by stronger modules.

## SPTM-style memory

Useful idea:

```text
observation retrieval -> topological node -> graph planning / waypoint hint
```

Implementation:

```text
NumpyCosineIndex
MemoryGraph.localize
MemoryGraph.get_revisit_candidates
MemoryGraph.score_candidate_exit
```

## Neural Topological SLAM-style topology plus coarse geometry

Useful idea:

```text
semantic node + coarse relative geometry edge
```

Implementation:

```text
MemoryNode.semantic
MemoryEdge.relative_pose_src_to_dst
edge covariance / uncertainty cost
```

## MapGPT-style promptable map context

Useful idea:

```text
summarize online map into LLM/VLM prompt
```

Implementation:

```text
MemoryGraph.build_vlm_memory_context
local_topology.candidate_exits
place_recognition.revisit_candidates
goal_context.goal_resume_hint
```

## PRISM-TopoMap-style no global node pose

Useful idea:

```text
locally aligned graph, no persistent global node coordinates
```

Implementation:

```text
node_global_pose_stored = False
relative edge transforms only
compose_path_pose for local subgoal reconstruction
```

## VLMaps / open-vocabulary map memory

Useful idea:

```text
language/object queryable spatial memory
```

Implementation:

```text
object_context
MemoryNode.semantic.object_belief
update_object_belief memory_op
replaceable image/text embedding hooks
```

## Structured ReAct / Graph-ReAct

Useful idea:

```text
observe -> diagnose -> act -> memory write
without free-form chain-of-thought
```

Implementation:

```text
prompts.py controlled reason-code prompt
NavMemoryAgent.step
memory_ops as verified requests
```

## Explicitly deferred

```text
pose graph optimization
```

The current version records relative constraints and merge/loop evidence. It does not solve the optimization problem. The TODO hook is present so this can be added cleanly later.
