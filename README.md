# voca-s2e
VOCA-style reasoning + S2E trajectory backbone

## Habitat + PixelNav + VLM Memory Integration

The current Habitat/PixelNav integration lives in:

```text
third_party/Pixel-Navigator/
```

This vendored directory contains the PixelNav runtime code plus the VOCA memory benchmark runner:

```text
third_party/Pixel-Navigator/voca_memory_benchmark.py
```

Core memory-side code is kept in:

```text
qwen_nav_memory_framework_v5/
goal_adapter/
```

See the vendored integration notes for setup and smoke commands:

```text
third_party/Pixel-Navigator/VOCA_INTEGRATION.md
```
