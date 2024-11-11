import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

class DDQNAgent:
    def __init__(self, state_size, action_size, batch_size=128, gamma=0.99,
                 lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_frequency=5000, memory_size=50000, model_path="agent_model.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.memory = deque(maxlen=memory_size)
        self.model_path = model_path

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        if reward >= 50 or done:
            for _ in range(5):  # Höhere Priorität für wichtige Übergänge
                self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        # Filter out invalid entries (e.g., None) in the batch
        batch = [entry for entry in batch if all(e is not None for e in entry)]
        if len(batch) == 0:
            return  # Skip if batch is empty

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(list(states))
        actions = torch.LongTensor(list(actions))
        rewards = torch.FloatTensor(list(rewards))
        next_states = torch.FloatTensor(list(next_states))
        dones = torch.FloatTensor(list(dones))
        
        # Q-Learning Update
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for exploration-exploitation balance
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def train(self):
        for _ in range(20):  # Train in multiple small batches
            self.replay()
        if len(self.memory) % self.target_update_frequency == 0:
            self.update_target_model()

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, self.model_path)
        print("Model and memory saved successfully.")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=self.memory.maxlen)
            print("Model and memory loaded successfully.")
