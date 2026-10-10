import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SafetyNavEnv(gym.Env):
    def __init__(self):
        super(SafetyNavEnv, self).__init__()
        # Observation: [x, y, target_x, target_y]
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        # Action: [dx, dy]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.state = None
        self.target = np.array([8.0, 8.0])
        
        # Hazard Zones (Circular)
        self.hazards = [
            {'center': np.array([5.0, 5.0]), 'radius': 1.5},
            {'center': np.array([2.0, 8.0]), 'radius': 1.0}
        ]

    def reset(self, seed=None):
        self.state = np.array([1.0, 1.0]) # Start at bottom left
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.target])

    def step(self, action):
        # Clip action and move
        action = np.clip(action, -1, 1)
        self.state = np.clip(self.state + action * 0.5, 0, 10)
        
        # Calculate Reward (Distance)
        dist = np.linalg.norm(self.state - self.target)
        reward = -dist 
        
        # Calculate Cost (Safety Violation)
        cost = 0
        in_hazard = False
        for h in self.hazards:
            if np.linalg.norm(self.state - h['center']) < h['radius']:
                cost = 1
                in_hazard = True
        
        # Done condition
        done = dist < 0.5
        
        return self._get_obs(), reward, done, False, {"cost": cost, "hazard": in_hazard}