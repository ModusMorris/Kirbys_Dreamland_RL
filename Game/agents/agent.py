# agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DDQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99,
                 lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_frequency=1000, memory_size=10000):
        self.state_size = state_size  # Dimension des Zustands
        self.action_size = action_size  # Anzahl der möglichen Aktionen
        self.batch_size = batch_size
        self.gamma = gamma  # Diskontfaktor
        self.lr = lr  # Lernrate
        self.epsilon = epsilon_start  # Startwert für epsilon im epsilon-greedy-Verfahren
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency  # Wie oft das Zielnetzwerk aktualisiert wird
        self.memory = deque(maxlen=memory_size)  # Replay-Memory

        # Q-Netzwerke
        self.policy_net = self.build_model()
        self.target_net = self.build_model()
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.steps_done = 0

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        # Epsilon-greedy Aktionenauswahl
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Nicht genug Erfahrungen zum Trainieren

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Aktionen als Index
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q-Werte für aktuelle Zustände
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Max Q-Werte für nächste Zustände aus dem Zielnetzwerk
        next_q_values = self.target_net(next_states).detach()
        next_actions = torch.argmax(self.policy_net(next_states), dim=1)
        next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Q-Learning Zielwert
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Verlustfunktion und Optimierung
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon reduzieren
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Zielnetzwerk aktualisieren
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_network()
