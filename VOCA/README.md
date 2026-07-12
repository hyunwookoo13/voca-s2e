# VOCA: Execution-Grounded Qwen-VLM Navigation

This directory contains the VOCA research runner used to study a local Qwen-VL
semantic supervisor over a frozen PixelNav execution skill. The supervisor
selects a backend-generated RGB waypoint or issues `rotate`,
`request_observation`, and `stop`. A v6 episodic graph records verified travel,
directional failures, deadlocks, revisits, room semantics, and target evidence.

The current standalone system uses PixelNav. The executor and memory contracts
are kept separate so S2E can replace PixelNav in the next integration stage.

## System Boundary

- Policy inputs: target category/context cues, RGB observations, declared
  localization, coarse-goal contract, executable candidates, and verified
  memory.
- Evaluation-only data: Habitat success, SPL, distance-to-goal, shortest paths,
  and top-down maps. These are never included in Qwen policy inputs.
- Every `go` must pass the candidate gate; ambiguous points use an independent
  RGB verifier; every `stop` uses target-completion verification.
- Current benchmark claims must be described as RGB plus declared simulator
  pose. Derived coarse-goal runs are not category-only ObjectNav results.

## Repository Layout

```text
VOCA/
  objnav_benchmark.py          # benchmark entry point
  qwen_vlm_planner.py          # Qwen supervisor and verification policy
  qwen_action_runner.py        # action/PixelNav closed loop
  pixel_candidate_gate.py      # executable RGB waypoint contract
  voca_memory_sidecar.py       # v6 memory adapter
  navigation_supervisor.py     # state and progress semantics
  qwen_nav_memory_framework_v6/# v6 memory development source
  llm_utils/                   # Qwen/Gemini/Ollama clients and priors
  scripts/                     # readiness, smoke, stress, and ablation tools
  tests/                       # unit and integration regression tests
  voca_s2e_bridge.py           # canonical v6 memory runtime adapter
  docs/                        # architecture specification
  reports/                     # curated reports and lightweight artifacts
```

Datasets, checkpoints, full benchmark outputs, caches, and temporary images are
intentionally excluded from Git.

## Environment

The validated local environment uses Python 3.9, Habitat-Lab 0.3.3,
Habitat-Sim, PyTorch, OpenCV, Transformers, and an OpenAI-compatible Qwen-VL
server. Install the lightweight Python dependencies with:

```bash
pip install -r requirements.txt
```

Habitat-Lab and Habitat-Sim must be installed with versions compatible with the
HM3D ObjectNav configuration.

## Local Assets

Configure these paths directly or place local links/files under `data/` and
`checkpoints/`:

```bash
export VOCA_SCENE_DATASETS_DIR=/path/to/scene_datasets
export VOCA_DATASETS_DIR=/path/to/episode_datasets
export VOCA_POLICY_CHECKPOINT=/path/to/pixelnav_A.ckpt
export VOCA_YOLOE_CHECKPOINT=/path/to/yoloe-11l-seg.pt
```

YOLOE is retained only for the original VOCA baseline. The default integrated
planner is YOLOE-free.

## Qwen Configuration

```bash
export VOCA_LLM_BACKEND=qwen
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_API_KEY=EMPTY
export QWEN_MODEL=qwen3-vl-32b-thinking-awq
export VOCA_QWEN_MODEL_ROOT=QuantTrio/Qwen3-VL-32B-Thinking-AWQ
```

`scripts/voca_env.sh` supplies the remaining validated defaults. Override any
value before invoking a run script.

## Readiness and Tests

```bash
source scripts/voca_env.sh

${HOME}/micromamba/bin/micromamba run \
  -r "${HOME}/micromamba-root" -n habitat \
  python scripts/check_benchmark_ready.py

${HOME}/micromamba/bin/micromamba run \
  -r "${HOME}/micromamba-root" -n habitat \
  python -m unittest discover -s tests
```

Asset-dependent setup tests skip cleanly when local HM3D data or checkpoints
are not installed.

## Benchmark Commands

One-episode integrated smoke:

```bash
VOCA_EVAL_EPISODES=1 scripts/run_objnav_smoke.sh
```

Bounded debugging run:

```bash
VOCA_EVAL_EPISODES=1 VOCA_MAX_EPISODE_STEPS=80 scripts/run_objnav_smoke.sh
```

Paired memory/candidate-scoring ablation:

```bash
VOCA_ABLATION_EPISODES=30 scripts/run_navigation_ablation_suite.sh
```

Each completed episode writes a metrics row, reproducibility manifest,
`qwen_calls.jsonl`, `steps.json`, memory graph, RGB/reasoning video, and
evaluation-only top-down video under the configured output directory.

## Curated Results

- `reports/weekend_20260711_12/`: implementation summary and presentation videos
- `reports/reproducibility_20260712/`: lightweight metrics, manifests, stress,
  calibration, model comparison, and ablation records
- `reports/qwen_vlm_model_comparison_report_20260710.md`: Qwen model comparison

The three-episode ablation is diagnostic only and does not establish
statistical superiority. Larger paired, multi-seed benchmarks are required for
paper-level performance claims.

## Development And Git Sync

The standalone VOCA checkout is the development source. After changing the
runner, visualizer, or v6 memory implementation, preview and apply the
one-way synchronization into the sibling `voca-s2e` Git checkout:

```bash
scripts/sync_to_voca_s2e.sh --dry-run
scripts/sync_to_voca_s2e.sh
```

The script mirrors the VOCA runner into `voca-s2e/VOCA/`, the memory package
into `voca-s2e/qwen_nav_memory_framework_v6/`, and the graph visualizer into
`voca-s2e/goal_adapter/`. Runtime data, checkpoints, full outputs, caches, and
local environment files are never copied.
