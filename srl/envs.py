"""Environment construction and the synthetic "human" preference signal.

The task reward of SafetyPointGoal1 only measures progress to the goal, and the
CMDP cost only fires *inside* a hazard. A human supervisor typically also cares
about a softer property the designer never wrote down: keeping a safety margin
around hazards instead of grazing their edge. We encode that hidden preference
as an oracle reward that is never shown to the agent:

    r_oracle(s, a) = r_task(s, a) - alpha * proximity(s')
    proximity(s')  = sum_h clip((m - d_h) / m, 0, 1),  d_h = dist(agent, hazard_h) - radius

The oracle is only used to answer pairwise preference queries (see
reward_model.py) and to evaluate how well each agent matches the supervisor.
"""

import numpy as np
import safety_gymnasium


class PreferenceOracleEnv:
    """Wraps a Safety-Gymnasium env and exposes the hidden oracle reward via `info`."""

    def __init__(self, env_id="SafetyPointGoal1-v0", margin=0.3, alpha=0.03):
        self.env = safety_gymnasium.make(env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.margin = margin
        self.alpha = alpha

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def _hazard_distances(self):
        task = self.env.unwrapped.task
        agent_xy = task.agent.pos[:2]
        hazards_xy = np.asarray(task.hazards.pos)[:, :2]
        return np.linalg.norm(hazards_xy - agent_xy, axis=1) - task.hazards.size

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        d = self._hazard_distances()
        proximity = np.clip((self.margin - d) / self.margin, 0.0, 1.0).sum()
        info["proximity"] = float(proximity)
        info["near_miss"] = float(np.any((d > 0) & (d < self.margin)))
        info["oracle_reward"] = float(reward - self.alpha * proximity)
        # goal bonus is +1, dense progress reward is O(1e-2) per step
        info["goal_reached"] = float(reward > 0.5)
        return obs, reward, cost, terminated, truncated, info
