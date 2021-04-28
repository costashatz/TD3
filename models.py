import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_size=256):
        super(ActorTD3, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

        self.max_action = torch.FloatTensor(max_action).to(device)
        self.min_action = torch.FloatTensor(min_action).to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return (self.max_action - self.min_action) * ((torch.tanh(self.l3(a)) + 1) / 2) + self.min_action


class ActorDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_sizes=[400, 300]):
        super(ActorDDPG, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)

        self.max_action = torch.FloatTensor(max_action).to(device)
        self.min_action = torch.FloatTensor(min_action).to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return (self.max_action - self.min_action) * ((torch.tanh(self.l3(a)) + 1) / 2) + self.min_action


class CriticDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[400, 300]):
        super(CriticDDPG, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0] + action_dim, hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DualCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DualCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class CriticWithOptimizer:
    def __init__(self, model, lr_critic=3e-4, critic_weight_decay=-1):
        self.model = model
        if critic_weight_decay > 0:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr_critic)

    def __call__(self, state, action):
        return self.model(state, action)

    def update(self, state, action, target_Q):
        # Get current Q estimate
        current_Q = self.model(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.optim.zero_grad()
        critic_loss.backward()
        self.optim.step()

        # print(current_Q.mean().item(), target_Q.mean().item())

    def parameters(self):
        return self.model.parameters()

    def to(self, dev):
        self.model.to(dev)

        return self


class DualCriticWithOptimizer:
    def __init__(self, model, lr_critic=3e-4, critic_weight_decay=-1):
        self.model = model
        if critic_weight_decay > 0:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr_critic)

    def __call__(self, state, action):
        return self.model(state, action)

    def Q1(self, state, action):
        return self.model.Q1(state, action)

    def update(self, state, action, target_Q):
        # Get current Q estimates
        current_Q1, current_Q2 = self.model(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.optim.zero_grad()
        critic_loss.backward()
        self.optim.step()

    def parameters(self):
        return self.model.parameters()

    def to(self, dev):
        self.model.to(dev)

        return self
