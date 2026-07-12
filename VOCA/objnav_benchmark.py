import os
import argparse
import csv
import cv2
import imageio
import numpy as np
import time
import random
from tqdm import tqdm
from settings import (
    DEFAULT_CUDA_VISIBLE_DEVICES,
    DEFAULT_DEVICE,
    DETECT_OBJECTS,
    OBJNAV_METRICS_PATH,
    POLICY_CHECKPOINT,
    TRAJECTORY_DIR,
    YOLOE_CHECKPOINT_PATH,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat
from habitat_config import hm3d_config
from vlm_planner import VLMPlanner
from qwen_point_planner import QwenPointPlanner, render_qwen_debug_frame
from qwen_vlm_planner import QwenVLMPlanner
from qwen_action_runner import run_qwen_vlm_action_episode
from coarse_goal_provider import CoarseGoalProvider
from pre_memory_readiness import write_pre_memory_readiness
from nav_audit import NavigationAuditLogger
from policy_agent import PolicyAgent
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from cv_utils.yoloe_detector import initialize_yoloe_model
from omegaconf import OmegaConf, open_dict


QWEN_PLANNERS = {"qwen_point", "qwen_vlm"}


def write_metrics(metrics, path):
    if not metrics:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)


def priors_context_item_count(priors):
    if not isinstance(priors, dict):
        return 0
    total = 0
    for key in ("Supports", "StrongCooccurs", "Gateways", "Lookalikes"):
        value = priors.get(key, [])
        if isinstance(value, (list, tuple, set)):
            total += len(value)
        elif value:
            total += 1
    return int(total)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=400)
    parser.add_argument("--stage", type=str, default="val")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--policy_checkpoint", type=str, default=POLICY_CHECKPOINT)
    parser.add_argument("--yoloe_checkpoint", type=str, default=YOLOE_CHECKPOINT_PATH)
    parser.add_argument("--output_dir", type=str, default=TRAJECTORY_DIR)
    parser.add_argument("--metrics_path", type=str, default=OBJNAV_METRICS_PATH)
    parser.add_argument("--planner", type=str, default="voca_yoloe", choices=["voca_yoloe", "qwen_point", "qwen_vlm"])
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=int(os.getenv("VOCA_BENCHMARK_SEED", "20260710")))
    parser.add_argument(
        "--episode_start_index",
        type=int,
        default=int(os.getenv("VOCA_EPISODE_START_INDEX", "0")),
    )
    parser.add_argument(
        "--episode_pool_size",
        type=int,
        default=int(os.getenv("VOCA_EPISODE_POOL_SIZE", "0")),
    )
    parser.add_argument(
        "--coarse_goal_file",
        type=str,
        default=os.getenv("VOCA_COARSE_GOAL_FILE", ""),
    )
    return parser.parse_known_args()[0]


args = get_args()
coarse_goal_provider = (
    CoarseGoalProvider.from_path(args.coarse_goal_file)
    if str(args.coarse_goal_file or "").strip()
    else None
)
os.environ["VOCA_BENCHMARK_SEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
try:
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
except Exception:
    pass

episode_start_index = max(0, int(args.episode_start_index))
episode_pool_size = max(
    episode_start_index + int(args.eval_episodes),
    int(args.episode_pool_size or 0),
)
habitat_config = hm3d_config(
    stage=args.stage,
    episodes=episode_pool_size,
    seed=args.seed,
    max_episode_steps=(
        int(args.max_episode_steps)
        if int(args.max_episode_steps or 0) > 0
        else None
    ),
)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)

OmegaConf.set_readonly(habitat_config, False)

from habitat.config.default_structured_configs import NumStepsMeasurementConfig
with open_dict(habitat_config.habitat.task.measurements):
    if "num_steps" not in habitat_config.habitat.task.measurements:
        habitat_config.habitat.task.measurements.num_steps = NumStepsMeasurementConfig()

habitat_env = habitat.Env(habitat_config)

if args.planner == "qwen_point":
    nav_planner = QwenPointPlanner()
elif args.planner == "qwen_vlm":
    nav_planner = QwenVLMPlanner()
else:
    yoloe_model = initialize_yoloe_model(
        weights=args.yoloe_checkpoint,
        device=args.device,
        classes=DETECT_OBJECTS,
        prompt_mode="text",
    )
    nav_planner = VLMPlanner(yoloe_model)

nav_executor = PolicyAgent(model_path=args.policy_checkpoint, device=args.device)
evaluation_metrics = []

for _ in range(episode_start_index):
    habitat_env.reset()

for i in tqdm(
    range(episode_start_index, episode_start_index + int(args.eval_episodes))
):
    if args.planner == "qwen_vlm":
        episode_metrics = run_qwen_vlm_action_episode(
            habitat_env,
            nav_planner,
            nav_executor,
            episode_index=i,
            output_dir=args.output_dir,
            max_episode_steps=args.max_episode_steps,
            topdown_fn=adjust_topdown,
            coarse_goal_provider=coarse_goal_provider,
        )
        evaluation_metrics.append(episode_metrics)
        write_metrics(evaluation_metrics, args.metrics_path)
        write_pre_memory_readiness(args.output_dir, args.metrics_path)
        continue

    obs = habitat_env.reset()

    # Wrap step to accumulate per-episode Euclidean travel distance.
    _stats = {
        "dist_m": 0.0,
        "prev": np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32),
    }
    _orig_step = habitat_env.step
    def _instrumented_step(action):
        obs_ = _orig_step(action)
        cur = np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32)
        _stats["dist_m"] += float(np.linalg.norm(cur - _stats["prev"]))
        _stats["prev"] = cur
        return obs_
    habitat_env.step = _instrumented_step

    trajectory_dir = os.path.join(args.output_dir, "trajectory_%d" % i)
    os.makedirs(trajectory_dir, exist_ok=True)
    audit_logger = NavigationAuditLogger() if args.planner in QWEN_PLANNERS else None
    audit_paths = {"steps_json": "", "memory_graph_json": ""}
    if hasattr(nav_planner, "set_qwen_call_log_path"):
        nav_planner.set_qwen_call_log_path(os.path.join(trajectory_dir, "qwen_calls.jsonl"))
    fps_writer = imageio.get_writer("%s/fps.mp4" % trajectory_dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % trajectory_dir, fps=4)
    heading_offset = 0
    step_counter = 0
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])
    prev_boxes = None
    curr_boxes = None
    pending_verify = False
    goal_flag = False
    deadlock_llm_calls = 0
    verification_llm_calls = 0
    truncated_by_max_steps = False

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]
    decision_by_frame = {}
    planning_frame_indices = set()

    def append_planning_frame(image):
        idx = len(episode_images)
        decision_by_frame[idx] = getattr(nav_planner, "last_decision", None)
        planning_frame_indices.add(idx)
        episode_images.append(image)

    def _agent_position_xyz():
        return [float(x) for x in habitat_env.sim.get_agent_state().position]

    def _metrics_snapshot():
        return dict(habitat_env.get_metrics())

    def record_qwen_audit(image, call_type, selected_view_id=None):
        if audit_logger is None:
            return
        decision = getattr(nav_planner, "last_decision", None)
        if not decision:
            return
        audit_logger.record_decision(
            decision=decision,
            target_object=habitat_env.current_episode.object_category,
            image_shape=image.shape,
            frame_index=len(episode_images) - 1,
            position_xyz=_agent_position_xyz(),
            metrics=_metrics_snapshot(),
            priors=getattr(nav_planner, "latest_priors", {}),
            angles=decision.get("angles"),
            call_type=call_type,
            selected_view_id=selected_view_id,
        )

    # Measure per-episode compute time (exclude video I/O).
    episode_t0 = time.perf_counter()

    nav_planner.query_priors_text()

    for _ in range(11):
        obs = habitat_env.step(3)
        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        step_counter += 1
    goal_image, goal_mask, debug_image, vis_rgb, goal_rotate,  pri_flag, obj_detected = nav_planner.make_plan(episode_images[-12:])
    pending_verify = (pri_flag and (not obj_detected))
    goal_flag = obj_detected
    for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            obs = habitat_env.step(2)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        step_counter += 1

    record_qwen_audit(goal_image, "make_plan_initial", selected_view_id=goal_rotate)
    append_planning_frame(vis_rgb)
    append_planning_frame(vis_rgb)
    nav_executor.reset(goal_image, goal_mask)


    while not habitat_env.episode_over:
        if args.max_episode_steps > 0 and int(habitat_env.get_metrics().get('num_steps', 0)) >= args.max_episode_steps:
            truncated_by_max_steps = True
            break
        action, skill_image = nav_executor.step(obs['rgb'], habitat_env.sim.previous_step_collided)

        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            step_counter += 1

        else:
            if habitat_env.episode_over:
                break
            for _ in range(0, abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset -= 1
                    step_counter += 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset += 1
                    step_counter += 1

            if pending_verify and action == 0:
                for _ in range(11):
                    if habitat_env.episode_over: break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                _llm_calls_before = int(nav_planner.llm_call_count)
                (
                    verify_goal_image,
                    verify_goal_mask,
                    verify_debug_image,
                    verify_vis,
                    verify_rotate,
                    verify_pri_flag,
                    verify_obj_detected
                ) = nav_planner.make_plan(episode_images[-12:])
                verification_llm_calls += int(nav_planner.llm_call_count) - _llm_calls_before

                for j in range(min(11 - verify_rotate, 1 + verify_rotate)):
                    if habitat_env.episode_over: break
                    if verify_rotate <= 6:
                        obs = habitat_env.step(3)
                    else:
                        obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                record_qwen_audit(verify_goal_image, "make_plan_verify", selected_view_id=verify_rotate)
                append_planning_frame(verify_vis)
                append_planning_frame(verify_vis)

                if verify_obj_detected:
                    goal_flag = True
                    pending_verify = False
                    continue

                elif verify_pri_flag:
                    pending_verify = True
                    goal_flag = False
                    prev_boxes = getattr(nav_planner, "_last_bboxes", [])
                else:
                    pending_verify = False
                    goal_flag = False
                    prev_boxes = getattr(nav_planner, "_last_bboxes", [])

                nav_executor.reset(verify_goal_image, verify_goal_mask)
                continue

            prev_boxes = getattr(nav_planner, "_last_bboxes", []) if prev_boxes is None else prev_boxes

            pano7 = []
            angles7 = []

            for _ in range(3):
                if habitat_env.episode_over: break
                obs = habitat_env.step(3)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                step_counter += 1

            pano7.append(episode_images[-1]); angles7.append(-90)

            for k in range(6):
                if habitat_env.episode_over: break
                obs = habitat_env.step(2)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                step_counter += 1
                pano7.append(obs['rgb'])
                angles7.append(-90 + 30*(k+1))  # -60, -30, 0, +30, +60, +90

            if habitat_env.episode_over or len(pano7) == 0:
                break

            (
                direction_image,   # goal_rgb
                debug_mask,
                pri_flag,
                obj_detected,
                debug_vis,         # vis_rgb
                curr_boxes,
                best_idx
            ) = nav_planner.apply_priors_on_image(pano7, return_boxes=True)


            if nav_planner.are_bboxes_similar(
                prev_boxes, curr_boxes,
                class_sensitive=False,
                ignore_classes=['floor','ground','flooring'],
                return_detail=False
            ):
                for _ in range(11):
                    if habitat_env.episode_over: break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                _llm_calls_before = int(nav_planner.llm_call_count)
                (
                    goal_image,
                    goal_mask,
                    debug_image2,
                    vis_rgb2,
                    goal_rotate,
                    pri_flag,
                    obj_detected
                ) = nav_planner.make_plan(episode_images[-12:])
                deadlock_llm_calls += int(nav_planner.llm_call_count) - _llm_calls_before

                for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
                    if habitat_env.episode_over: break
                    if goal_rotate <= 6:
                        obs = habitat_env.step(3)
                    else:
                        obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                record_qwen_audit(goal_image, "make_plan_deadlock", selected_view_id=goal_rotate)
                append_planning_frame(vis_rgb2); append_planning_frame(vis_rgb2)

                prev_boxes = getattr(nav_planner, "_last_bboxes", [])
                pending_verify = (pri_flag and (not obj_detected))
                goal_flag = obj_detected

            else:
                cur_deg = int(angles7[-1])           # +90
                sel_deg = int(angles7[best_idx])     # {-90,-60,-30,0,30,60,90}
                delta = sel_deg - cur_deg
                turns = abs(delta) // 30
                if delta < 0:
                    for _ in range(turns):
                        if habitat_env.episode_over: break
                        obs = habitat_env.step(3)
                        episode_images.append(obs['rgb'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        step_counter += 1
                elif delta > 0:
                    for _ in range(turns):
                        if habitat_env.episode_over: break
                        obs = habitat_env.step(2)
                        episode_images.append(obs['rgb'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        step_counter += 1

                record_qwen_audit(direction_image, "apply_priors_on_image", selected_view_id=best_idx)
                append_planning_frame(debug_vis); append_planning_frame(debug_vis)
                goal_image, goal_mask = direction_image, debug_mask
                pending_verify = pri_flag and (not obj_detected)
                goal_flag = obj_detected
                prev_boxes = curr_boxes

            print("action", action)
            print("goal _flag", goal_flag)
            print("step_counter", step_counter)
            step_counter = 0
            nav_executor.reset(goal_image, goal_mask)

    habitat_env.step = _orig_step

    episode_t1 = time.perf_counter()
    episode_time_sec = episode_t1 - episode_t0


    active_decision = None
    for idx, img in enumerate(episode_images):
        if idx in decision_by_frame:
            active_decision = decision_by_frame[idx]
        if args.planner in QWEN_PLANNERS:
            fps_writer.append_data(
                render_qwen_debug_frame(
                    img,
                    decision=active_decision,
                    target=habitat_env.current_episode.object_category,
                    planner_name=args.planner,
                    draw_point=idx in planning_frame_indices,
                )
            )
        else:
            fps_writer.append_data(img)

    for top in episode_topdowns:
        topdown_writer.append_data(top)

    fps_writer.close()
    topdown_writer.close()

    if hasattr(nav_planner, "save_qwen_calls"):
        nav_planner.save_qwen_calls(os.path.join(trajectory_dir, "qwen_calls.jsonl"))

    if audit_logger is not None:
        audit_logger.finalize_pending(
            position_xyz=_agent_position_xyz(),
            metrics=_metrics_snapshot(),
            collision=bool(getattr(habitat_env.sim, "previous_step_collided", False)),
        )
        audit_paths = audit_logger.save_run(os.path.join(trajectory_dir, "memory"))

    qwen_calls = getattr(nav_planner, "qwen_call_log", [])
    qwen_last = qwen_calls[-1] if qwen_calls else {}
    qwen_point = qwen_last.get("Point") if isinstance(qwen_last.get("Point"), list) else [None, None]
    latest_priors = getattr(nav_planner, "latest_priors", {})

    evaluation_metrics.append({
        'episode': i,
        'planner_name': args.planner,
        'object_goal': habitat_env.current_episode.object_category,
        'success': habitat_env.get_metrics()['success'],
        'spl': habitat_env.get_metrics()['spl'],
        'start_distance_to_goal': start_geodesic_m,
        'final_distance_to_goal': habitat_env.get_metrics()['distance_to_goal'],
        'llm_calls': int(nav_planner.llm_call_count),
        'llm_calls_deadlock': int(deadlock_llm_calls),
        'llm_calls_verification': int(verification_llm_calls),
        'llm_success_calls': int(nav_planner.llm_success_count),
        'llm_error_calls': int(nav_planner.llm_error_count),
        'llm_avg_time_sec': float(np.mean(nav_planner.llm_durations)) if len(nav_planner.llm_durations) > 0 else 0.0,
        'priors_calls': int(len(getattr(nav_planner, 'priors_durations', []))),
        'priors_avg_time_sec': float(np.mean(getattr(nav_planner, 'priors_durations', []))) if getattr(nav_planner, 'priors_durations', []) else 0.0,
        'qwen_navigation_vlm_calls': int(len(getattr(nav_planner, 'navigation_vlm_durations', []))),
        'qwen_navigation_vlm_avg_time_sec': float(np.mean(getattr(nav_planner, 'navigation_vlm_durations', []))) if getattr(nav_planner, 'navigation_vlm_durations', []) else 0.0,
        'llm_last_error': str(nav_planner.llm_last_error) if nav_planner.llm_last_error else "",
        'priors_success_count': int(getattr(nav_planner, "priors_success_count", 0)),
        'priors_parse_fail_count': int(getattr(nav_planner, "priors_parse_fail_count", 0)),
        'priors_fallback_count': int(getattr(nav_planner, "priors_fallback_count", 0)),
        'priors_context_item_count': priors_context_item_count(latest_priors),
        'priors_last_error': str(getattr(nav_planner, "priors_last_error", "")) if getattr(nav_planner, "priors_last_error", "") else "",
        'vlm_json_fallback_count': int(getattr(nav_planner, "vlm_json_fallback_count", 0)),
        'vlm_json_last_error': str(getattr(nav_planner, "vlm_json_last_error", "")) if getattr(nav_planner, "vlm_json_last_error", "") else "",
        'qwen_point_calls': int(len(qwen_calls)),
        'qwen_point_fallback_count': int(sum(1 for call in qwen_calls if call.get("fallback"))),
        'qwen_last_angle': qwen_last.get("Angle", ""),
        'qwen_last_u': qwen_point[0],
        'qwen_last_v': qwen_point[1],
        'qwen_last_confidence': qwen_last.get("Confidence", ""),
        'qwen_last_reason': qwen_last.get("Reason", ""),
        'steps_json': audit_paths.get('steps_json', ""),
        'memory_graph_json': audit_paths.get('memory_graph_json', ""),
        'yoloe_detect_calls': int(len(nav_planner.yoloe_durations)),
        'yoloe_detect_avg_time_sec': float(np.mean(nav_planner.yoloe_durations)) if len(nav_planner.yoloe_durations) > 0 else 0.0,
        'yoloe_detect_total_time_sec': float(np.sum(nav_planner.yoloe_durations)) if len(nav_planner.yoloe_durations) > 0 else 0.0,
        'episode_time_sec': float(episode_time_sec),
        'truncated_by_max_steps': int(truncated_by_max_steps),
        'num_steps': int(habitat_env.get_metrics().get('num_steps', 0)),
        'total_distance_m': float(_stats['dist_m']),
    })

    write_metrics(evaluation_metrics, args.metrics_path)
