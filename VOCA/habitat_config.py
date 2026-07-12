import os

import habitat
from settings import (
    EPISODE_PREFIX,
    HM3D_CONFIG_PATH,
    MP3D_CONFIG_PATH,
    SCENE_PREFIX,
)
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.config.default_structured_configs import LookUpActionConfig,LookDownActionConfig,NumStepsMeasurementConfig

def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


# habitat config used to run the benchmark for objnav in hm3d dataset
def hm3d_config(
    path: str = HM3D_CONFIG_PATH,
    stage: str = "val",
    episodes=200,
    seed=20260710,
    success_distance=None,
    allow_sliding=None,
    max_episode_steps=None,
):
    if success_distance is None:
        success_distance = float(os.environ.get("VOCA_HM3D_SUCCESS_DISTANCE_M", "0.10"))
    if allow_sliding is None:
        allow_sliding = _env_bool("VOCA_HM3D_ALLOW_SLIDING", False)
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENE_PREFIX
        habitat_config.habitat.dataset.data_path = EPISODE_PREFIX + "objectnav/hm3d/v2/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = SCENE_PREFIX + "hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        if max_episode_steps is not None and int(max_episode_steps) > 0:
            habitat_config.habitat.environment.max_episode_steps = int(
                max_episode_steps
            )
        habitat_config.habitat.seed = int(seed)
        habitat_config.habitat.simulator.seed = int(seed)
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.simulator.forward_step_size=0.25
        habitat_config.habitat.simulator.turn_angle=30
        habitat_config.habitat.task.measurements.success.success_distance = float(
            success_distance
        )
        habitat_config.habitat.simulator.habitat_sim_v0.allow_sliding = bool(
            allow_sliding
        )
    return habitat_config

# habitat config used to run the benchmark for objnav in mp3d dataset
def mp3d_config(path:str=MP3D_CONFIG_PATH,stage:str='val',episodes=200,seed=20260710):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENE_PREFIX
        habitat_config.habitat.dataset.data_path = EPISODE_PREFIX + "objectnav/mp3d/v1/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = SCENE_PREFIX + "mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.seed = int(seed)
        habitat_config.habitat.simulator.seed = int(seed)
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.simulator.forward_step_size=0.25
        habitat_config.habitat.simulator.turn_angle=30
        habitat_config.habitat.task.measurements.success.success_distance = 1.0
        habitat_config.habitat.simulator.habitat_sim_v0.allow_sliding = True
    return habitat_config

# habitat config used to generate the pixel-nav training data in hm3d scenes
def hm3d_data_config(path:str=HM3D_CONFIG_PATH,
                     stage:str='val',
                     episodes=100,
                     robot_height=0.88,
                     robot_radius=0.25,
                     sensor_height=0.88,
                     image_width=640,
                     image_height=480,
                     image_hfov=79,
                     step_size=0.25,
                     turn_angle=30):

    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENE_PREFIX
        habitat_config.habitat.dataset.data_path = EPISODE_PREFIX + "objectnav/hm3d/v2/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = SCENE_PREFIX + "hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.simulator.agents.main_agent.height=robot_height
        habitat_config.habitat.simulator.agents.main_agent.radius=robot_radius
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 50.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth =0.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.simulator.forward_step_size = step_size
        habitat_config.habitat.simulator.turn_angle = turn_angle
        habitat_config.habitat.task.measurements = {'num_steps':NumStepsMeasurementConfig}
        habitat_config.habitat.task.actions['look_up'] = LookUpActionConfig()
        habitat_config.habitat.task.actions['look_down'] = LookDownActionConfig()
    return habitat_config

# habitat config used to generate the pixel-nav training data in mp3d scenes
def mp3d_data_config(path:str=MP3D_CONFIG_PATH,
                     stage:str='val',
                     episodes=200,
                     robot_height=0.88,
                     robot_radius=0.25,
                     sensor_height=0.88,
                     image_width=640,
                     image_height=480,
                     image_hfov=79,
                     step_size=0.25,
                     turn_angle=30):

    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = SCENE_PREFIX
        habitat_config.habitat.dataset.data_path = EPISODE_PREFIX + "objectnav/mp3d/v1/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = SCENE_PREFIX + "mp3d/mp3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.simulator.agents.main_agent.height=robot_height
        habitat_config.habitat.simulator.agents.main_agent.radius=robot_radius

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 50.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.simulator.forward_step_size = step_size
        habitat_config.habitat.simulator.turn_angle = turn_angle
        habitat_config.habitat.task.measurements = {'num_steps':NumStepsMeasurementConfig}
        habitat_config.habitat.task.actions['look_up'] = LookUpActionConfig()
        habitat_config.habitat.task.actions['look_down'] = LookDownActionConfig()
    return habitat_config
