# Architecture Notes v0.5

## Design claim

The memory is a **relative-pose topo-metric episodic memory graph with directional negative edges**.

It is topological because nodes represent places/situations and graph edges represent connectivity. It is topo-metric because edges also store relative SE(2) transforms. It does not store global node poses.

## Closed-loop control contract

```text
coarse goal + RGB + memory
  -> Qwen semantic supervisor
  -> go / rotate / request_observation / stop
```

When `go` is selected, the output contains `selected_view_type`, `selected_image_point`, and `fine_goal`. That is the only information the fast S2E/PixelNav-style module needs. When the VLM cannot safely choose a waypoint, it requests additional views or rotation.

## Rotate-to-front policy

`NavAgentConfig.force_front_view_waypoint=True` enables the safest deployment policy.

```text
Qwen selects left/right/back fine goal
  -> backend rotates toward that view
  -> backend obtains fresh front observation
  -> Qwen is called again
  -> fast module receives a front-view fine goal
```

This is useful when the low-level navigation skill only accepts front-camera waypoints.

## VLM-grounded place recognition

The backend does not make final semantic same-place decisions by itself.

```text
backend embedding index
  -> top-k revisit candidates
  -> Qwen compares current image with memory image/summary
  -> Qwen emits confirm_revisit_node or request_merge_nodes
  -> backend verifies confidence, retrieval score, node state, and critical-memory conflicts
  -> backend commits or rejects
```

False merges are treated as worse than false negatives. The prompt tells Qwen to prefer `request_observation` when uncertain.

## Soft same-place merge policy

A revisit can have a non-trivial baseline or parallax relative to the first visit. Therefore the default merge policy is **soft merge** rather than hard deletion.

```text
first/canonical node is preserved
revisit node is preserved
T_canonical_to_revisit is computed by composing graph edges when possible
canonical -> revisit edge_type = revisit_link
revisit -> canonical edge_type = revisit_link_reverse
image paths, keyframe ids, and descriptions are shared through same_place_clusters
constraint edges are excluded from local_topology.candidate_exits
```

This gives Qwen common place memory while keeping the geometry of each baseline available for later reasoning.

## Spatial plausibility gate

Visual place recognition can suffer from aliasing in repeated corridors, similar hotel rooms, or symmetric layouts. v0.5 therefore adds a backend gate before `confirm_revisit_node` or `request_merge_nodes` is committed.

```text
VLM says: same_place candidate
Backend computes: T_candidate_to_current by relative-pose chain composition
Backend checks: baseline distance, graph hops, uncertainty, visual score, VLM confidence
Accept: plausible parallax/baseline
Reject: distant visual alias or no connected relative path
```

Default global upper bounds are configurable in `NavAgentConfig`:

```python
revisit_max_same_place_baseline_m = 3.0
revisit_extended_same_place_baseline_m = 5.0
revisit_max_same_place_hops = 12
```

The backend then applies stricter place-category gates. For example, corridors, doorways, and unknown places are more likely to be visually aliased, so their effective same-place baseline is smaller than an intersection or a distinctive room. If the pair looks similar but is beyond the same-place gate, the context may expose `same_region_allowed=true`, but the backend still rejects the same-place soft merge.

False-positive merge prevention is the priority. Duplicate nodes are acceptable; corrupting the graph with a wrong same-place link is not.

This is not pose graph optimization. It does not globally correct edge constraints; it only decides whether a proposed same-place link is locally plausible.

## Deadlock handling

A deadlock is represented at two levels:

```text
node.navigation_state.deadlock_status:
  none -> suspected -> confirmed -> escaping -> confirmed_escaped

edge.traversal.status:
  deadlock_entry_candidate
  deadlock_entry
  escape_success
```

This prevents over-forbidding an entire room. The incoming edge is the primary negative memory; the reverse/backtrack edge can be a valid escape edge.

## Goal preservation

Goal memory is kept outside the graph in the episode-level `goal_context` generated each step. A detour can be labeled as `escaping_deadlock` while the coarse goal bearing and distance remain visible to Qwen.

## Image storage policy

Images are active evidence, not permanent prompt memory.

```text
HOT:
  active keyframes may be attached to VLM prompt

COMPRESSED:
  active image_ref is removed from prompt context
  embedding, summary, relative edges, negative memory remain

ARCHIVED / TOMBSTONE:
  merge history is retained for auditability
```

This mirrors SLAM marginalization conceptually, but without running a SLAM optimizer.

## ObjNav hook

ObjNav is supported through:

```text
CoarseGoal.from_object_goal
memory.object_context
MemoryNode.semantic.object_belief
memory_ops update_object_belief / mark_object_seen
```

The object belief is a soft prior. Qwen must still choose visible navigable floor or request more observations.


## Runtime current-pose relation to latest node

The graph does not store global node poses, but the closed-loop agent needs to know where the robot currently is relative to the latest node anchor. v0.5 therefore maintains a live floating transform:

```text
T_latest_node_to_current_robot
```

This transform is derived from robot backend state or odometry and is not persisted as a node global pose. It is used to transform outgoing graph edges into the current robot frame, expose `memory.current_pose_relation_to_latest_node` to Qwen, and form cumulative relative edges when a new node is committed.

Pose graph optimization remains a TODO; this live relation is only a local runtime transform.

## Pose graph optimization TODO

The framework stores loop closure and merge evidence, but does not optimize the pose graph.

```python
MemoryGraph.optimize_pose_graph()
# raises NotImplementedError
```

This is intentional for v0.5. Future work can add GTSAM/g2o-style optimization or a project-specific relative constraint solver.
