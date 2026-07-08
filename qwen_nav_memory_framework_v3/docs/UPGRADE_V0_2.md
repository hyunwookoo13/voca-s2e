# Upgrade v0.2 Summary

Implemented after design review:

```text
1. VLM-grounded place recognition verifier
2. Backend-verified revisit commit and merge request handling
3. Rotate-to-front fine-goal policy
4. Deadlock suspected/confirmed/escaping/escaped state machine
5. Directional negative-edge guard before executing go
6. ObjNav object-belief memory hook
7. Prompt update for place recognition, memory_ops, and deadlock policy
8. Tests for upgraded behavior
9. Pose graph optimization explicitly deferred as TODO
```

The schema envelope remains `nav_vlm_waypoint_v1`; memory context is extended to `nav_memory_context_v4`.


Note: v0.3 adds runtime current-pose relation to the latest node and robot-frame exit scoring.
