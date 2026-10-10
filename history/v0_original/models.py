import torch
import torch.nn as nn
import torch.optim as optim

# --- RLHF Reward Model ---
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --- Hierarchical Agent ---
class HRLAgent:
    def __init__(self, device):
        self.device = device
        
        # Manager: State -> Subgoal
        self.manager = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)
        
        # Worker: State + Subgoal -> Action
        self.worker = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)
        
        # Lagrangian Multiplier (Learnable Safety Parameter)
        self.log_lagrange = torch.nn.Parameter(torch.zeros(1, requires_grad=True).to(device))
        
        self.optimizer = optim.Adam(
            list(self.manager.parameters()) + 
            list(self.worker.parameters()) + 
            [self.log_lagrange], lr=0.0003
        )

    def select_action(self, state_t, subgoal_t):
        # Input: Tensors on Device
        # Output: Tensor Action Mean
        inputs = torch.cat([state_t, subgoal_t], dim=1)
        return self.worker(inputs)

    def select_subgoal(self, state_t):
        return self.manager(state_t)