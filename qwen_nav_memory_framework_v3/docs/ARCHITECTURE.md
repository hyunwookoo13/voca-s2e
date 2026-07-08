# Architecture Notes v0.3

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

The graph does not store global node poses, but the closed-loop agent needs to know where the robot currently is relative to the latest node anchor. v0.3 therefore maintains a live floating transform:

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

This is intentional for v0.3. Future work can add GTSAM/g2o-style optimization or a project-specific relative constraint solver.
