# Reference Idea Map

This document explains which external research ideas informed the framework.

## SPTM

Useful ideas:

- non-parametric topological memory graph
- observation-to-node retrieval network
- graph planning over stored places
- memory subsampling
- temporal edges and shortcut/loop edges
- lookahead waypoint rather than full-path low-level control

Framework implementation:

- `NumpyCosineIndex` for current observation to keyframe retrieval
- `MemoryGraph` nodes/edges for places/connectivity
- `score_candidate_exit` and `compose_path_pose` for route hints
- node creation thresholds and compression policy

## Neural Topological SLAM

Useful ideas:

- semantic features on nodes
- approximate/coarse geometric reasoning through edges
- robust operation under noisy actuation

Framework implementation:

- `MemoryNode.semantic`
- `MemoryEdge.relative_pose_src_to_dst`
- edge covariance and uncertainty cost

## MapGPT

Useful ideas:

- online map summarized into the LLM/VLM prompt
- node information plus topological relationships
- adaptive subgoal planning rather than single reactive choice

Framework implementation:

- `build_vlm_memory_context`
- `local_topology.candidate_exits`
- `goal_context.goal_resume_hint`
- compact `retrieved_memory_images`

## PRISM-TopoMap

Useful ideas:

- graph of locally aligned locations
- no reliance on global metric node coordinates
- online localization and loop closure with place recognition

Framework implementation:

- no global node pose stored
- relative edge transforms only
- localization through visual index
- merge hook in `merge_nodes`

## VLMaps and open-vocabulary map memory

Useful ideas:

- visual-language features can make map memory queryable by language/object goals
- object/room priors help ObjNav

Framework implementation:

- semantic summary field
- object belief hook
- replaceable embedding model

## Structured ReAct / Graph-ReAct

Useful idea:

- interleave observation, diagnosis, action, and memory write
- avoid uncontrolled chain-of-thought by using reason codes

Framework implementation:

- `prompts.py` system prompt
- `reasoning.decision_reason` and `reasoning.goal_reason`
- optional verified `memory_ops`
