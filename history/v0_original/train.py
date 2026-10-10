import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
import numpy as np

# Local Imports
from env import SafetyNavEnv
from models import HRLAgent, RewardModel

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on {device} ---")

    env = SafetyNavEnv()
    agent = HRLAgent(device)
    reward_model = RewardModel(input_dim=6).to(device)

    # Hyperparameters
    episodes = 1000
    max_steps = 200
    gamma = 0.99
    cost_limit = 0.1

    pbar = tqdm(range(episodes), desc="Training")
    moving_avg_reward = 0

    # 2. Training Loop
    for ep in pbar:
        obs, _ = env.reset()
        done = False
        traj_data = []
        
        # Manager selects sub-goal
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        sub_goal = agent.select_subgoal(obs_tensor) 
        
        step_count = 0
        while not done and step_count < max_steps:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Worker Action
            action_mean = agent.select_action(obs_t, sub_goal)
            action = action_mean.cpu().detach().numpy()[0] + np.random.normal(0, 0.1, size=2)
            
            next_obs, _, done, _, info = env.step(action)
            
            # RLHF Reward Query
            with torch.no_grad():
                act_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                rlhf_reward = reward_model(obs_t, act_t).item()

            traj_data.append({
                's': obs_t,
                'a': torch.FloatTensor(action).unsqueeze(0).to(device),
                'r': rlhf_reward,
                'c': info['cost'],
                'sub_goal': sub_goal
            })
            
            obs = next_obs
            step_count += 1

        # 3. Update Step
        R = 0
        returns = []
        total_cost = 0
        
        for step in reversed(traj_data):
            R = step['r'] + gamma * R
            returns.insert(0, R)
            total_cost += step['c']
        
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        policy_loss = []
        avg_cost = total_cost / len(traj_data)
        
        for i, step in enumerate(traj_data):
            # Re-compute action for gradient
            action_pred = agent.select_action(step['s'], step['sub_goal'])
            dist = Normal(action_pred, 0.1)
            log_prob = dist.log_prob(step['a']).sum()
            
            lagrange_val = F.softplus(agent.log_lagrange)
            advantage = returns[i]
            
            # Loss = -LogProb * (Reward - Lambda * (Cost - Limit))
            step_loss = -log_prob * (advantage - lagrange_val.detach() * (avg_cost - cost_limit))
            policy_loss.append(step_loss)

        lagrange_loss = -agent.log_lagrange * (avg_cost - cost_limit)

        agent.optimizer.zero_grad()
        final_loss = torch.stack(policy_loss).sum() + lagrange_loss
        final_loss.backward()
        agent.optimizer.step()

        # Logging
        ep_reward = sum([t['r'] for t in traj_data])
        moving_avg_reward = 0.05 * ep_reward + 0.95 * moving_avg_reward
        
        if ep % 10 == 0:
            pbar.set_postfix({
                'Rew': f"{moving_avg_reward:.2f}",
                'Cost': f"{avg_cost:.2f}",
                'Lmb': f"{F.softplus(agent.log_lagrange).item():.2f}"
            })

if __name__ == "__main__":
    train()