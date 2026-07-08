# Architecture Notes

## Design claim

The memory is a **relative-pose topo-metric episodic memory graph with directional negative edges**.

It is topological because nodes represent places/situations and graph edges represent connectivity. It is topo-metric because edges also store relative SE(2) transforms. It does not store global node poses.

## Deadlock handling

A deadlock is represented at two levels:

1. Node state: the current place is suspected/confirmed deadlock.
2. Directed edge state: the entry edge into this place is a failed/deadlock branch.

This avoids over-forbidding an entire room when only one entry direction or view sector is problematic.

```json
{
  "edge_id": "e_hall_room",
  "traversal": {
    "status": "deadlock_entry",
    "failure_count": 1,
    "escape_edge_id": "e_room_hall"
  }
}
```

## Goal preservation

Goal memory is kept outside the graph in the episode-level `goal_context` generated each step. Therefore a detour can be labeled as `escaping_deadlock` while the coarse goal bearing and distance remain visible to Qwen.

## Image storage policy

Images are active evidence, not permanent prompt memory.

- Hot keyframes can be attached to the VLM prompt.
- Compressed nodes drop `image_ref` from VLM-facing context.
- Embeddings, summaries, relative edge constraints, and negative memories remain.

This is analogous to SLAM marginalization: old visual evidence leaves the active prompt window while constraints remain in the graph.

## Backend responsibilities

The VLM should not be trusted to directly update graph topology. The backend should verify:

- visual similarity
- odometry consistency
- collision/progress outcome
- current topological neighborhood

before applying merge, deadlock, or compression decisions.

## Extension hooks

For ObjNav, add this to `MemoryNode.semantic`:

```json
{
  "object_belief": {
    "target_object": "mug",
    "seen_target": false,
    "candidate_objects": [
      {"label": "cup", "confidence": 0.41, "bbox_px": [221, 188, 270, 250]}
    ],
    "room_object_prior": {"kitchen": 0.72, "office": 0.38}
  }
}
```

For larger deployments, replace the default embedding/indexing with:

- DINOv2 / SigLIP / CLIP / Qwen image encoder
- FAISS HNSW/IVF-PQ or hnswlib
- optional pose graph optimizer for loop closure constraints
