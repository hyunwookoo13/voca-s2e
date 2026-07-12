import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VOCA_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = VOCA_ROOT.parent
SIBLING_VOCA_S2E_ROOT = PARENT_ROOT / "voca-s2e"
LOCAL_NAV_MEMORY_ROOT = VOCA_ROOT / "qwen_nav_memory_framework_v6"
if (PARENT_ROOT / "qwen_nav_memory_framework_v6").exists():
    DEFAULT_VOCA_S2E_ROOT = PARENT_ROOT
elif (SIBLING_VOCA_S2E_ROOT / "qwen_nav_memory_framework_v6").exists():
    DEFAULT_VOCA_S2E_ROOT = SIBLING_VOCA_S2E_ROOT
else:
    DEFAULT_VOCA_S2E_ROOT = PARENT_ROOT
VOCA_S2E_ROOT = Path(
    os.environ.get("VOCA_S2E_ROOT", str(DEFAULT_VOCA_S2E_ROOT))
).expanduser().resolve()
MONOREPO_NAV_MEMORY_ROOT = VOCA_S2E_ROOT / "qwen_nav_memory_framework_v6"
if LOCAL_NAV_MEMORY_ROOT.exists():
    DEFAULT_NAV_MEMORY_ROOT = LOCAL_NAV_MEMORY_ROOT
else:
    DEFAULT_NAV_MEMORY_ROOT = MONOREPO_NAV_MEMORY_ROOT
NAV_MEMORY_QWEN_ROOT = Path(
    os.environ.get(
        "VOCA_S2E_NAV_MEMORY_ROOT",
        str(DEFAULT_NAV_MEMORY_ROOT),
    )
).expanduser().resolve()


@dataclass(frozen=True)
class NavMemoryModules:
    agent: Any
    embedding: Any
    memory_graph: Any
    policy: Any
    robot_backend: Any
    schema: Any
    safety: Any
    utils: Any
    vlm_client: Any


def ensure_voca_s2e_on_path() -> Path:
    if not VOCA_S2E_ROOT.exists():
        raise FileNotFoundError("voca-s2e repository not found: {}".format(VOCA_S2E_ROOT))
    root_text = str(VOCA_S2E_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return VOCA_S2E_ROOT


def ensure_nav_memory_qwen_on_path() -> Path:
    if not NAV_MEMORY_QWEN_ROOT.exists():
        raise FileNotFoundError(
            "voca-s2e nav memory framework not found: {}".format(NAV_MEMORY_QWEN_ROOT)
        )
    root_text = str(NAV_MEMORY_QWEN_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return NAV_MEMORY_QWEN_ROOT


def import_memory_graph_visualizer() -> Any:
    try:
        return importlib.import_module("memory_graph_visualizer")
    except ImportError:
        ensure_voca_s2e_on_path()
        return importlib.import_module("goal_adapter.memory_graph_visualizer")


def import_nav_memory_qwen() -> NavMemoryModules:
    ensure_nav_memory_qwen_on_path()
    return NavMemoryModules(
        agent=importlib.import_module("nav_memory_qwen.agent"),
        embedding=importlib.import_module("nav_memory_qwen.embedding"),
        memory_graph=importlib.import_module("nav_memory_qwen.memory_graph"),
        policy=importlib.import_module("nav_memory_qwen.policy"),
        robot_backend=importlib.import_module("nav_memory_qwen.robot_backend"),
        schema=importlib.import_module("nav_memory_qwen.schema"),
        safety=importlib.import_module("nav_memory_qwen.safety"),
        utils=importlib.import_module("nav_memory_qwen.utils"),
        vlm_client=importlib.import_module("nav_memory_qwen.vlm_client"),
    )
