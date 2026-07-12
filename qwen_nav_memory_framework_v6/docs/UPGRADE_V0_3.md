# Upgrade v0.3 Summary

Added after the latest-node pose review:

```text
1. Runtime current-pose relation to latest/current node
2. Transient node-anchor robot state in NavMemoryAgent
3. memory.current_pose_relation_to_latest_node in VLM context
4. Candidate exits transformed from node frame to current robot frame
5. New temporal edges use cumulative latest-node-to-robot pose
6. Tests for live pose update, cumulative edge creation, and robot-frame exit bearing
7. Pose graph optimization remains TODO and is still not executed
```

Important distinction:

```text
Stored in graph edges:
  relative_pose_src_to_dst

Runtime only, not a global node pose:
  T_latest_node_to_current_robot
```

This keeps the memory topological/topo-metric while making the current robot position usable for VLM action selection and graph-based subgoal reconstruction.
