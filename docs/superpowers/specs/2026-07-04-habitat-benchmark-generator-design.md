# Habitat Benchmark Generator Design

## Goal

Generate S2E-free VLM decision benchmark cases from multiple Habitat-Sim scenes using pathfinder/navmesh metadata instead of hand-written smoke cases.

## Scope

This generator creates case JSON files and RGB assets compatible with `goal_adapter.vlm_benchmark`. It supports the four validation categories:

- `coarse_gps`
- `coarse_object_point`
- `tracking_loss`
- `deadlock`

It does not run S2E end-to-end. It prepares a dataset that can support a 4 x 100 case report.

## Data Flow

```text
scene paths
 -> Habitat-Sim simulator
 -> pathfinder navigable point sampling
 -> RGB render at sampled robot pose
 -> candidate waypoint metadata
 -> VLM benchmark case JSON
 -> existing VLM benchmark runner
 -> RGB/top-down overlay reports
```

## Label Strategy

The first implementation uses navmesh/pathfinder labels:

- `coarse_gps`: choose a coarse target and label the best local navigable candidate as `expected.goal_xy`.
- `coarse_object_point`: use a synthetic object-point proxy and label an object-front navigable floor candidate.
- `tracking_loss`: remove target point and label expected action as `LOOK_AROUND`.
- `deadlock`: mark one local goal as failed/blocked and label a different candidate as `RESELECT_GOAL`.

Semantic ObjectNav labels can be added later when semantic annotations are available and reliable.

## Output Layout

```text
<output-dir>/
  assets/
    <case_id>_rgb.png
  0001_<case_id>.json
  0002_<case_id>.json
```

Each case JSON uses the existing schema:

```json
{
  "case_id": "...",
  "category": "...",
  "input": {... GoalAdapterInput ...},
  "expected": {... benchmark expectation ...}
}
```

## Success Criteria

- Unit tests can generate all four categories using a fake Habitat-Sim module.
- Generated cases can be loaded by `load_vlm_benchmark_cases`.
- Running the existing benchmark on generated cases writes overlay reports.
- The CLI can generate a small real Habitat-Sim smoke dataset from an HM3D scene.
