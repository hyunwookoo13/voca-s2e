import torch
import cv2
import numpy as np
import quaternion
import os
import time
from typing import Any, Dict, List, Sequence
from settings import DEFAULT_DEVICE, POLICY_CHECKPOINT
from policy_network import PixelNavPolicy

class PolicyAgent:
    def __init__(self,model_path=POLICY_CHECKPOINT,max_token_length=64,image_size=224,device=DEFAULT_DEVICE):
        self.image_size = image_size
        self.max_token_length = max_token_length
        self.device = device
        self.network = PixelNavPolicy(max_token_length,device)
        self.network.load_state_dict(torch.load(model_path,map_location=device))
        self.network.eval()
    def reset(self,goal_image,goal_mask):
        self.history_image = np.zeros(
            (self.max_token_length,self.image_size,self.image_size,3),
            dtype=np.uint8,
        )
        self.goal_image = cv2.resize(cv2.cvtColor(goal_image,cv2.COLOR_BGR2RGB),(self.image_size,self.image_size))
        self.goal_image = self.goal_image[np.newaxis,:,:,:]
        self.goal_mask = cv2.resize(goal_mask,(self.image_size,self.image_size),cv2.INTER_NEAREST)
        self.goal_mask = self.goal_mask[np.newaxis,:,:,np.newaxis]
        self.predict_length = 0
        self.collide_times = 0
        self.collide_action = 0

    def step(self,obs_image,collide=False,early_stop=True):
        if self.predict_length >= self.max_token_length:
            raise RuntimeError(
                "PixelNav history exhausted: {} observations for max_token_length={}".format(
                    self.predict_length + 1,
                    self.max_token_length,
                )
            )
        self.current_obs = cv2.resize(cv2.cvtColor(obs_image,cv2.COLOR_BGR2RGB),(self.image_size,self.image_size))
        self.history_image[self.predict_length] = self.current_obs.copy() #append(self.current_obs)
        # The decoder is causally masked, so zero-padded future frames cannot affect
        # the current action. Encoding only the observed prefix preserves the output
        # while avoiding a 64-frame CNN pass at every navigation step.
        self.input_image = self.history_image[np.newaxis,:self.predict_length + 1,:,:,:]
        with torch.inference_mode():
            action_pred,distance_pred,goal_pred = self.network(self.goal_mask,self.goal_image,self.input_image)
        action_logits = action_pred[0][self.predict_length].detach().clone()
        if collide == False:
            return_action = action_logits.cpu().numpy().argmax()
        else:
            action_logits[1] = -1
            return_action = action_logits.cpu().numpy().argmax()
        return_distance = distance_pred[0][self.predict_length].detach().cpu().numpy()
        return_goal = np.array(obs_image)
        goal_mask = goal_pred[0][self.predict_length].detach().cpu().numpy()
        goal_y = goal_mask[0] * obs_image.shape[0]
        goal_x = goal_mask[1] * obs_image.shape[1]
        return_goal = cv2.rectangle(return_goal,(int(goal_x)-5,int(goal_y)-5),(int(goal_x)+5,int(goal_y)+5),(255,0,0),-1)
        return_goal = cv2.putText(return_goal,"%d"%int(return_distance[0]*10),(310,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6, cv2.LINE_AA)
        self.predict_length += 1
        if early_stop and self.predict_length > 32:
            return_action = 0
        return return_action,cv2.cvtColor(return_goal,cv2.COLOR_BGR2RGB)

    def score_pixel_candidates(
        self,
        rgb_images: Sequence[np.ndarray],
        candidates: Sequence[Dict[str, Any]],
        *,
        point_radius: int = 8,
    ) -> List[Dict[str, Any]]:
        """Estimate candidate executability with PixelNav without changing agent state."""
        prepared: List[Dict[str, Any]] = []
        goal_images: List[np.ndarray] = []
        goal_masks: List[np.ndarray] = []
        observations: List[np.ndarray] = []
        image_by_view_id = {index: np.asarray(image) for index, image in enumerate(rgb_images)}

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                view_id = int(candidate.get("view_id", 0))
                point = candidate.get("point_px")
                image = image_by_view_id[view_id]
                u, v = int(point[0]), int(point[1])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if image.ndim != 3 or image.shape[2] != 3:
                continue
            height, width = image.shape[:2]
            u = max(0, min(width - 1, u))
            v = max(0, min(height - 1, v))
            mask = np.zeros((height, width), dtype=np.uint8)
            radius = max(1, int(point_radius))
            cv2.rectangle(
                mask,
                (max(0, u - radius), max(0, v - radius)),
                (min(width - 1, u + radius), min(height - 1, v + radius)),
                255,
                -1,
            )
            processed = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                (self.image_size, self.image_size),
            )
            goal_images.append(processed)
            observations.append(processed.copy())
            goal_masks.append(
                cv2.resize(mask, (self.image_size, self.image_size), cv2.INTER_NEAREST)[
                    :, :, np.newaxis
                ]
            )
            prepared.append(candidate)

        if not prepared:
            return []

        goal_image_batch = np.stack(goal_images, axis=0)
        goal_mask_batch = np.stack(goal_masks, axis=0)
        episode_batch = np.stack(observations, axis=0)[:, np.newaxis, :, :, :]
        try:
            batch_size = max(1, int(os.environ.get("VOCA_PIXELNAV_SCORE_BATCH_SIZE", "6")))
        except ValueError:
            batch_size = 6
        probability_batches: List[np.ndarray] = []
        distance_batches: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                end = min(len(prepared), start + batch_size)
                action_pred, distance_pred, _goal_pred = self.network(
                    goal_mask_batch[start:end],
                    goal_image_batch[start:end],
                    episode_batch[start:end],
                )
                probability_batches.append(
                    torch.softmax(action_pred[:, 0, :], dim=-1).detach().cpu().numpy()
                )
                distance_batches.append(
                    distance_pred[:, 0, 0].detach().cpu().numpy()
                )
        probabilities = np.concatenate(probability_batches, axis=0)
        distances = np.concatenate(distance_batches, axis=0)

        results: List[Dict[str, Any]] = []
        action_names = ["stop", "forward", "turn_left", "turn_right", "look_up", "look_down"]
        normalizer = float(np.log(max(2, len(action_names))))
        for candidate, probs, distance_raw in zip(prepared, probabilities, distances):
            clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
            entropy = float(-np.sum(clipped * np.log(clipped)))
            confidence = max(0.0, min(1.0, 1.0 - entropy / normalizer))
            locomotion_probability = float(np.sum(clipped[1:4]))
            stop_probability = float(clipped[0])
            horizontal_probability = stop_probability + locomotion_probability
            feasibility = locomotion_probability * (0.65 + 0.35 * confidence)
            predicted_action_id = int(np.argmax(clipped))
            results.append(
                {
                    "candidate_ref": str(candidate.get("candidate_ref") or ""),
                    "schema_version": "pixelnav_candidate_feasibility_v1",
                    "available": True,
                    "policy_feasibility_score": round(float(feasibility), 4),
                    "predicted_action_id": predicted_action_id,
                    "predicted_action": action_names[predicted_action_id],
                    "action_confidence": round(float(confidence), 4),
                    "horizontal_action_probability": round(horizontal_probability, 4),
                    "locomotion_probability": round(locomotion_probability, 4),
                    "stop_probability": round(stop_probability, 4),
                    "vertical_action_probability": round(float(np.sum(clipped[4:])), 4),
                    "predicted_distance_raw": round(float(distance_raw), 4),
                    "predicted_distance_m_estimate": round(max(0.0, float(distance_raw) * 10.0), 4),
                    "action_probabilities": {
                        name: round(float(probability), 4)
                        for name, probability in zip(action_names, clipped)
                    },
                }
            )
        return results
