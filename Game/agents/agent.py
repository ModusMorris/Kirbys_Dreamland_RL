import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class DDQNAgent:
    def __init__(self, state_size, action_size, batch_size=256, gamma=0.97,
                 lr=1e-4, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999999975,
                 target_update_frequency=5000, memory_size=500000, model_path="agent_model.pth", writer=None, **kwargs):
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
        self.writer = writer  # TensorBoard writer for logging metrics

        # Verwende die GPU, wenn verfügbar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialisiere das Modell und das Zielmodell auf dem richtigen Gerät (CPU oder GPU)
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_model()

    def _build_model(self):
        # Berechne die Ausgabegröße der Convolutional Layers, um sie für den Fully Connected Layer zu nutzen
        #conv_output_size = self._get_conv_output_size()

        # Modelldefinition mit Convolutional Layers
        model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 8, 128),  # Angepasste Fully-Connected-Schicht
            nn.ReLU(),
            nn.Linear(128, self.action_size)
                )
        return model
        
        
        # #kleineres CNN Model 
        # model = nn.Sequential(
        #     nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(conv_output_size, 128),  # Weniger Neuronen
        #     nn.ReLU(),
        #     nn.Linear(128, self.action_size)
        # )
        # return model
    
    # def _get_conv_output_size(self):
    #     with torch.no_grad():
    #         conv_layers = nn.Sequential(
    #             nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
    #             nn.ReLU(),
    #             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    #             nn.ReLU(),
    #             nn.Flatten(),
    #             nn.Linear(64 * 10 * 8, 128),  # Angepasste Fully-Connected-Schicht
    #             nn.ReLU(),
    #             nn.Linear(128, self.action_size)
    #         ).to(self.device)
    #         input_size = (1, 4, 20, 16)  # Stapel von 4 Frames
    #         dummy_input = torch.zeros(input_size).to(self.device)
    #         output = conv_layers(dummy_input)
    #         return output.numel()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def replay(self, current_epoch):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(list(actions)).to(self.device)
        rewards = torch.FloatTensor(list(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(list(dones)).to(self.device)

        # Double Q-Learning Update
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TensorBoard Logging
        if self.writer:
            self.writer.add_scalar("Loss", loss.item(), current_epoch)

        # Update Epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def train(self, current_epoch):
        for _ in range(5):  # Trainiere in kleineren Batches
            self.replay(current_epoch)

        # Update Target Model Periodically
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
        print(f"Model weights saved successfully to {checkpoint_path}.")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', self.optimizer.state_dict()))
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.memory = deque(checkpoint.get('memory', []), maxlen=self.memory.maxlen)
            print("Model and memory loaded successfully.")
