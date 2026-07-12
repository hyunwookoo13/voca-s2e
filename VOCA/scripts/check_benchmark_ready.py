#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ok(label: str, detail: str = "") -> None:
    print(f"OK {label}{(': ' + detail) if detail else ''}")


def _fail(label: str, detail: str) -> None:
    raise RuntimeError(f"{label}: {detail}")


def _check_file(label: str, path: str) -> None:
    p = Path(path)
    if not p.is_file():
        _fail(label, f"missing file {p}")
    _ok(label, str(p))


def _check_path(label: str, path: str) -> None:
    p = Path(path)
    if not p.exists():
        _fail(label, f"missing path {p}")
    _ok(label, str(p))


def check_paths(planner: str) -> None:
    import settings

    _check_file("hm3d_config", settings.HM3D_CONFIG_PATH)
    _check_file("mp3d_config", settings.MP3D_CONFIG_PATH)
    _check_path("scene_datasets", settings.SCENE_PREFIX)
    _check_path("episode_datasets", settings.EPISODE_PREFIX)
    _check_file(
        "hm3d_objnav_val",
        str(Path(settings.EPISODE_PREFIX) / "objectnav/hm3d/v2/val/val.json.gz"),
    )
    _check_file(
        "hm3d_scene_dataset",
        str(Path(settings.SCENE_PREFIX) / "hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"),
    )
    _check_file("policy_checkpoint", settings.POLICY_CHECKPOINT)
    if planner == "voca_yoloe":
        _check_file("yoloe_checkpoint", settings.YOLOE_CHECKPOINT_PATH)


def check_imports(planner: str) -> None:
    import cv2
    import habitat
    import imageio
    import numpy
    import omegaconf
    import requests
    import torch

    _ok("import cv2", getattr(cv2, "__version__", ""))
    _ok("import habitat", getattr(habitat, "__version__", ""))
    _ok("import imageio", getattr(imageio, "__version__", ""))
    _ok("import numpy", getattr(numpy, "__version__", ""))
    _ok("import omegaconf", getattr(omegaconf, "__version__", ""))
    _ok("import requests", getattr(requests, "__version__", ""))
    _ok("import torch", getattr(torch, "__version__", ""))
    if planner == "voca_yoloe":
        from ultralytics import YOLOE

        _ok("import YOLOE", str(YOLOE))


def _served_models(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else []
    return [item for item in data or [] if isinstance(item, dict)]


def check_qwen() -> None:
    import requests

    base_url = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    expected_model = os.getenv("QWEN_MODEL", "").strip()
    expected_root = os.getenv("VOCA_QWEN_MODEL_ROOT", "").strip()
    if not expected_model:
        _fail("qwen_model", "QWEN_MODEL must name the exact served model id")
    if not expected_root:
        _fail(
            "qwen_model_root",
            "VOCA_QWEN_MODEL_ROOT must name the exact checkpoint root",
        )
    response = requests.get(f"{base_url}/models", timeout=10)
    response.raise_for_status()
    models = _served_models(response.json())
    served_ids = sorted(str(item.get("id") or "") for item in models)
    matching = [item for item in models if str(item.get("id") or "") == expected_model]
    if len(matching) != 1:
        _fail(
            "qwen_model",
            "expected {!r}, served ids={}".format(expected_model, served_ids),
        )
    served_root = str(matching[0].get("root") or "").strip()
    if served_root != expected_root:
        _fail(
            "qwen_model_root",
            "expected {!r}, served {!r}".format(expected_root, served_root),
        )
    _ok(
        "qwen_model",
        "id={} root={} base_url={}".format(
            expected_model,
            expected_root,
            base_url,
        ),
    )


def check_protocol_contract() -> None:
    from coarse_goal_provider import CoarseGoalProvider

    protocol = os.getenv("VOCA_BENCHMARK_PROTOCOL", "").strip()
    coarse_goal_path = os.getenv("VOCA_COARSE_GOAL_FILE", "").strip()
    if not protocol:
        _fail("benchmark_protocol", "VOCA_BENCHMARK_PROTOCOL is required")

    is_derived_protocol = "derived_coarse_goalnav" in protocol.lower()
    if coarse_goal_path and not is_derived_protocol:
        _fail(
            "benchmark_protocol",
            "a coarse-goal provider cannot be reported as category-only ObjectNav",
        )
    if is_derived_protocol and not coarse_goal_path:
        _fail(
            "benchmark_protocol",
            "derived CoarseGoalNav requires VOCA_COARSE_GOAL_FILE",
        )
    if not coarse_goal_path:
        _ok("benchmark_protocol", protocol)
        return

    provider = CoarseGoalProvider.from_path(coarse_goal_path)
    if not provider.records:
        _fail("coarse_goal_provider", "provider contains no records")
    expected_sha = os.getenv("VOCA_COARSE_GOAL_EXPECTED_SHA256", "").strip()
    if expected_sha and provider.source_sha256 != expected_sha:
        _fail(
            "coarse_goal_provider_sha256",
            "expected {}, actual {}".format(
                expected_sha,
                provider.source_sha256,
            ),
        )
    sources = sorted(
        {str(record.get("source") or "") for record in provider.records}
    )
    _ok(
        "coarse_goal_provider",
        "path={} records={} sha256={} sources={} reporting=derived-only".format(
            provider.source_path,
            len(provider.records),
            provider.source_sha256,
            sources,
        ),
    )
    _ok("benchmark_protocol", protocol)


def check_habitat_config_contract() -> None:
    from habitat_config import hm3d_config

    try:
        max_steps = int(os.getenv("VOCA_MAX_EPISODE_STEPS", "1000"))
    except ValueError:
        _fail("max_episode_steps", "VOCA_MAX_EPISODE_STEPS must be an integer")
    if max_steps <= 0:
        _fail("max_episode_steps", "VOCA_MAX_EPISODE_STEPS must be positive")
    config = hm3d_config(
        stage="val",
        episodes=1,
        max_episode_steps=max_steps,
    )
    configured = int(config.habitat.environment.max_episode_steps)
    if configured != max_steps:
        _fail(
            "max_episode_steps",
            "requested {}, Habitat configured {}".format(max_steps, configured),
        )
    _ok("max_episode_steps", str(configured))


def check_localization_contract() -> None:
    from navigation_supervisor import build_localization_contract

    source = os.getenv("VOCA_LOCALIZATION_SOURCE", "none")
    allow_sim = os.getenv("VOCA_ALLOW_SIM_POSE_POLICY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    contract = build_localization_contract(source=source, allow_sim_pose=allow_sim)
    if contract["source"] == "none" and os.getenv("VOCA_QWEN_MEMORY_SIDECAR", "1") != "0":
        _fail(
            "localization_contract",
            "memory benchmark must explicitly declare a localization source",
        )
    _ok("localization_contract", str(contract))


def check_habitat_reset() -> None:
    import habitat
    from habitat_config import hm3d_config

    max_steps = int(os.getenv("VOCA_MAX_EPISODE_STEPS", "1000"))
    config = hm3d_config(
        stage="val",
        episodes=1,
        max_episode_steps=max_steps,
    )
    env = habitat.Env(config)
    try:
        obs = env.reset()
        _ok(
            "hm3d_objnav_reset",
            f"episode={env.current_episode.episode_id} object={env.current_episode.object_category} obs={sorted(obs.keys())}",
        )
    finally:
        env.close()


def check_model_loads(device: str, planner: str) -> None:
    from policy_agent import PolicyAgent
    import settings

    if planner == "voca_yoloe":
        from cv_utils.yoloe_detector import initialize_yoloe_model

        initialize_yoloe_model(
            weights=settings.YOLOE_CHECKPOINT_PATH,
            device=device,
            classes=["bed", "chair", "floor"],
        )
        _ok("yoloe_load", device)

    PolicyAgent(model_path=settings.POLICY_CHECKPOINT, device=device)
    _ok("policy_load", device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether VOCA ObjectNav benchmark dependencies are ready.")
    parser.add_argument("--device", default=os.getenv("VOCA_DEVICE", "cuda:0"))
    parser.add_argument("--planner", default=os.getenv("VOCA_PLANNER", "qwen_vlm"))
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-env-reset", action="store_true")
    parser.add_argument("--skip-model-loads", action="store_true")
    args = parser.parse_args()

    check_paths(args.planner)
    check_imports(args.planner)
    check_localization_contract()
    check_protocol_contract()
    check_habitat_config_contract()
    if not args.skip_qwen:
        check_qwen()
    if not args.skip_env_reset:
        check_habitat_reset()
    if not args.skip_model_loads:
        check_model_loads(args.device, args.planner)
    _ok("benchmark_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
