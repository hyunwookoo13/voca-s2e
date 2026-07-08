# Upgrade v0.4 Summary

Added after the soft-merge / parallax review:

```text
1. Soft same-place merge is now the default revisit commit policy.
2. Revisit nodes are preserved instead of archived or hard-deleted.
3. Canonical and revisit nodes are linked by revisit_link / revisit_link_reverse relative constraint edges.
4. The canonical->revisit relative pose is computed by graph chain-rule composition when a path exists.
5. relative_pose_between_nodes() supports arbitrary connected node pairs, reverse-edge traversal, and JSON summaries.
6. same_place_clusters share image paths, keyframe ids, descriptions, and canonical/member node ids.
7. Constraint-only same-place edges are excluded from VLM candidate exits, but available for spatial reconstruction.
8. Pose graph optimization remains TODO and is still not executed.
```

The key distinction is now:

```text
Hard merge:
  destructive, redirects edges, archives one node. Still available as backend-only merge_nodes().

Soft merge:
  default, preserves both nodes, links them with relative constraints, shares place memory.
```
