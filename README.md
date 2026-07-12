# voca-s2e

Research workspace for execution-grounded VLM navigation with episodic memory.
The current benchmark system uses PixelNav as the motion backbone; S2E is the
planned fast-navigation backbone after the VLM/memory controller is stabilized.

## Repository layout

```text
VOCA/                           Git mirror of the Qwen-VLM + PixelNav runner
qwen_nav_memory_framework_v6/  Git mirror of the v6 episodic-memory package
goal_adapter/                   memory and goal visualization adapters
third_party/Pixel-Navigator/    earlier PixelNav integration and experiments
qwen_nav_memory_framework_v5/  retained legacy implementation
papers/                         research design notes and methodology
```

The active closed loop is:

```text
coarse goal -> Qwen VLM -> action/fine goal -> safety gate -> PixelNav
            -> execution audit -> v6 memory update -> next decision
```

`VOCA/voca_s2e_bridge.py` resolves the mirrored v6 package directly from this
repository. Set `VOCA_S2E_NAV_MEMORY_ROOT` only when testing another checkout.

## Development workflow

The standalone sibling `VOCA` checkout is the development source. Do not make
independent feature edits in this repository's `VOCA/` mirror. From the source
checkout, synchronize runner, memory, and visualizer changes with:

```bash
scripts/sync_to_voca_s2e.sh --dry-run
scripts/sync_to_voca_s2e.sh
```

After synchronization, run the checks below and commit from this Git checkout.

## Verification

```bash
python -m unittest discover -s qwen_nav_memory_framework_v6/tests -v
cd VOCA
python -m unittest discover -s tests -v
```

Runtime datasets, HM3D scenes, model weights, and generated benchmark outputs
remain local and are excluded from Git. See `VOCA/README.md` for setup and
benchmark commands. Curated summaries and presentation videos are under
`VOCA/reports/`.
