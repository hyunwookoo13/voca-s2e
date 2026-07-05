# VLM Benchmark Visualization Design

## Goal

Build per-case visual audit artifacts for the VLM decision benchmark so each result explains the situation, the selected RGB point, the robot pose, and the expected/predicted world goals before S2E integration.

## Scope

This spec covers visualization and report integration for the existing S2E-free VLM decision benchmark. It does not implement full ObjectNav end-to-end S2E evaluation. It prepares the report format needed for a later 400-case Habitat-Sim benchmark.

## Case Artifacts

Each benchmark case report should include:

- `current_rgb_overlay.png`: the current RGB frame with expected and predicted image points marked.
- `topdown_overlay.png`: a lightweight top-down map with robot pose, heading, candidate waypoints, coarse target, expected goal, predicted goal, and blocked/failed goals.
- `index.html`: an audit page showing RGB overlay, top-down overlay, pass/fail checks, expected output, parsed decision, and raw VLM response.

## Data Sources

The visualization uses fields already present in case input and expected output:

- `input.current_rgb`
- `input.current_pose`
- `input.heading`
- `input.current_goal_xy`
- `input.high_level_target.coarse_goal_xy`
- `input.high_level_target.coarse_image_point`
- `input.memory_summary.candidate_waypoints[*].goal_xy`
- `input.memory_summary.candidate_waypoints[*].image_point`
- `input.memory_summary.failed_goal_xy`
- `expected.goal_xy`
- parsed VLM `refined_goal_xy`
- parsed VLM `selected_image_point`

## Visual Encoding

RGB overlay markers:

- Blue: predicted VLM selected image point.
- Green: expected image point from expected goal/candidate when available.
- Orange: coarse object/image point when available.
- Red: blocked or failed point if represented in image coordinates.

Top-down overlay markers:

- Black circle and arrow: robot pose and heading.
- Gray: candidate waypoints.
- Orange: coarse target or current goal.
- Green: expected refined goal.
- Blue: predicted refined goal.
- Red: failed/blocked goals.

## Rendering Approach

Use a small pure-Python raster renderer instead of adding a runtime dependency on Pillow in the default Python environment. The renderer will support RGB/RGBA PNG files written by the existing Habitat smoke pipeline and generate simple PNG overlays with circles, arrows, lines, and labels.

## Success Criteria

- Existing benchmark tests still pass.
- A new visualization test can generate `current_rgb_overlay.png` and `topdown_overlay.png` from a synthetic case and decision.
- Running the VLM benchmark writes both overlay files for every case.
- Per-case HTML displays both overlay images when present.
- The generated report remains useful when some metadata is missing, such as tracking-loss cases without goal points.
