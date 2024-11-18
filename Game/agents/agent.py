import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import numpy as np
import threading

class CombinedModel(nn.Module):
    def __init__(self, frame_shape, game_area_shape, action_size):
        super(CombinedModel, self).__init__()
        # CNN für Frames
        self.conv_frames = nn.Sequential(
            nn.Conv2d(frame_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully Connected Layer für Game Area
        self.fc_game_area = nn.Sequential(
            
            nn.Linear(game_area_shape[0] * game_area_shape[1], 128),  # Eingabedimension berechnet aus game_area_shape
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Kombinierte Features
        combined_input_size = self._get_conv_output_size(frame_shape) + 128
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, frames, game_area):
        x1 = self.conv_frames(frames)
        game_area = game_area.view(game_area.size(0), -1)  # Flatten game_area
        #print(f"Flattened Game Area Shape: {game_area.shape}")  # Debugging
        x2 = self.fc_game_area(game_area)
        #print(f"Frames Output Shape: {x1.shape}, Game Area Output Shape: {x2.shape}")  # Debugging
        x = torch.cat([x1, x2], dim=1)
        return self.fc_combined(x)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_frames(dummy_input)
            return output.numel()


class DDQNAgent:
    def __init__(self, frame_shape, game_area_shape, action_size, batch_size=32, gamma=0.99,
                 lr=1e-3, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999,
                 target_update_frequency=7000, memory_size=500, model_path="agent_model.pth", writer=None, **kwargs):
        self.frame_shape = frame_shape
        self.game_area_shape = game_area_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.memory = deque(maxlen=memory_size)
        self.model_path = model_path
        self.writer = writer  # TensorBoard writer for logging metrics

        # Verwende die GPU, wenn verfügbar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialisiere das Modell und das Zielmodell auf dem richtigen Gerät (CPU oder GPU)
        self.model = CombinedModel(frame_shape, game_area_shape, action_size).to(self.device)
        self.target_model = CombinedModel(frame_shape, game_area_shape, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            frames = torch.FloatTensor(state["frames"]).unsqueeze(0).to(self.device)
            game_area = torch.FloatTensor(state["game_area"]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.model(frames, game_area)).item()

    def replay(self, current_epoch):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        frames, game_areas, actions, rewards, next_frames, next_game_areas, dones = zip(*[
            (item["frames"], item["game_area"], action, reward, next_item["frames"], next_item["game_area"], done)
            for item, action, reward, next_item, done in self.memory
        ])

        frames = torch.FloatTensor(np.array(frames)).to(self.device)
        game_areas = torch.FloatTensor(np.array(game_areas)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_frames = torch.FloatTensor(np.array(next_frames)).to(self.device)
        next_game_areas = torch.FloatTensor(np.array(next_game_areas)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-Learning Update
        q_values = self.model(frames, game_areas).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_frames, next_game_areas).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Optimizer Schritt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar("Loss", loss.item(), current_epoch)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def train(self, current_epoch):
        for _ in range(10):
            self.replay(current_epoch)
        if len(self.memory) % self.target_update_frequency == 0:
            self.update_target_model()

    def save_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.model_path

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, checkpoint_path)
        print(f"Model weights and optimizer state saved successfully to {checkpoint_path}.")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.memory = deque(checkpoint.get('memory', []), maxlen=self.memory.maxlen)
            print("Model and memory loaded successfully.")
        else:
            print("No checkpoint found. Starting with a new model.")
