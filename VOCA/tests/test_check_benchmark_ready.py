import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_benchmark_ready


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class BenchmarkReadyContractTests(unittest.TestCase):
    def test_qwen_check_requires_exact_id_and_checkpoint_root(self):
        response = _FakeResponse(
            {
                "data": [
                    {
                        "id": "qwen3-vl-32b-thinking-awq",
                        "root": "QuantTrio/Qwen3-VL-32B-Thinking-AWQ",
                    }
                ]
            }
        )
        env = {
            "QWEN_BASE_URL": "http://localhost:8000/v1",
            "QWEN_MODEL": "qwen3-vl-32b-thinking-awq",
            "VOCA_QWEN_MODEL_ROOT": "QuantTrio/Qwen3-VL-32B-Thinking-AWQ",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "requests.get",
            return_value=response,
        ):
            check_benchmark_ready.check_qwen()

        wrong_root = dict(env)
        wrong_root["VOCA_QWEN_MODEL_ROOT"] = "Qwen/Qwen3-VL-32B-Thinking"
        with patch.dict(os.environ, wrong_root, clear=True), patch(
            "requests.get",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "qwen_model_root"):
                check_benchmark_ready.check_qwen()

    def test_coarse_provider_requires_derived_protocol_and_optional_hash_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "coarse.json"
            payload = {
                "schema_version": "voca_coarse_goal_dataset_v1",
                "records": [
                    {
                        "episode_index": 0,
                        "map_xy": [1.0, 2.0],
                        "source": "s2e_backbone",
                        "type": "map_waypoint",
                        "uncertainty": "medium",
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            valid_env = {
                "VOCA_BENCHMARK_PROTOCOL": (
                    "hm3d_derived_coarse_goalnav_v1_sim_pose_noise0p5"
                ),
                "VOCA_COARSE_GOAL_FILE": str(path),
                "VOCA_COARSE_GOAL_EXPECTED_SHA256": digest,
            }
            with patch.dict(os.environ, valid_env, clear=True):
                check_benchmark_ready.check_protocol_contract()

            mislabeled = dict(valid_env)
            mislabeled["VOCA_BENCHMARK_PROTOCOL"] = (
                "hm3d_standard_metrics_sim_pose_declared_v1"
            )
            with patch.dict(os.environ, mislabeled, clear=True):
                with self.assertRaisesRegex(RuntimeError, "category-only"):
                    check_benchmark_ready.check_protocol_contract()

            wrong_hash = dict(valid_env)
            wrong_hash["VOCA_COARSE_GOAL_EXPECTED_SHA256"] = "0" * 64
            with patch.dict(os.environ, wrong_hash, clear=True):
                with self.assertRaisesRegex(RuntimeError, "sha256"):
                    check_benchmark_ready.check_protocol_contract()

    def test_derived_protocol_requires_provider(self):
        with patch.dict(
            os.environ,
            {
                "VOCA_BENCHMARK_PROTOCOL": (
                    "hm3d_derived_coarse_goalnav_v1_sim_pose_noise0p5"
                ),
                "VOCA_COARSE_GOAL_FILE": "",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires"):
                check_benchmark_ready.check_protocol_contract()

    def test_habitat_contract_passes_requested_max_steps(self):
        config = SimpleNamespace(
            habitat=SimpleNamespace(
                environment=SimpleNamespace(max_episode_steps=777)
            )
        )
        with patch.dict(
            os.environ,
            {"VOCA_MAX_EPISODE_STEPS": "777"},
            clear=True,
        ), patch("habitat_config.hm3d_config", return_value=config) as hm3d:
            check_benchmark_ready.check_habitat_config_contract()

        hm3d.assert_called_once_with(
            stage="val",
            episodes=1,
            max_episode_steps=777,
        )


if __name__ == "__main__":
    unittest.main()
