import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import numpy as np

class DDQNAgent:
    def __init__(self, state_size, action_size, batch_size=128, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995,
                 target_update_frequency=1000, memory_size=500000, model_path="agent_model.pth", writer=None, **kwargs):
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
        conv_output_size = self._get_conv_output_size()

        # Modelldefinition mit Convolutional Layers
        model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),  # Größere Fully Connected Layer
            nn.ReLU(),
            nn.Linear(256, self.action_size)  # Ausgabe: Anzahl der möglichen Aktionen
        )
        return model

    def _get_conv_output_size(self):
        with torch.no_grad():
            # Definiere die Convolutional Layers separat, damit wir die Dimension berechnen können
            conv_layers = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ).to(self.device)

            # Berechne die Ausgabegröße nach den Convolutional Layers
            input_size = (1, 4, 20, 16)  # Stapel von 4 Frames (Batch, Channel, Height, Width)
            dummy_input = torch.zeros(input_size).to(self.device)
            output = conv_layers(dummy_input)
            return output.numel()


    def _forward_conv(self, x):
        # Teile des Convolutional Layers
        conv_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        conv_layers = conv_layers.to(self.device)
        return conv_layers(x)

    def update_target_model(self):
        # Aktualisiere die Parameter des Zielmodells mit denen des Hauptmodells
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Speichere die Erfahrungen im Replay-Speicher
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        # Wähle eine Aktion entweder zufällig (Exploration) oder basierend auf dem Modell (Exploitation)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Zustand in einen FloatTensor konvertieren und auf die GPU verschieben
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

        # Q-Learning Update
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Optimizer Schritt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Protokolliere den Verlust ins TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("Loss", loss.item(), current_epoch)

        # Update von epsilon für den Exploration-Exploitations-Ausgleich
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def train(self, current_epoch):
        # Trainiere das Modell in kleineren Batches
        for _ in range(10):
            self.replay(current_epoch)

        # Aktualisiere das Zielmodell in regelmäßigen Abständen
        if len(self.memory) % (self.target_update_frequency) == 0:
            self.update_target_model()

    def save_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.model_path

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer-Zustand hinzufügen
            'epsilon': self.epsilon,
            'memory': list(self.memory)  # Replay-Speicher optional speichern
        }, checkpoint_path)
        print(f"Model weights saved successfully to {checkpoint_path}.")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Lade die Modellspezifikationen
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

            # Überprüfen, ob der Optimizer-Zustand vorhanden ist
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Warning: 'optimizer_state_dict' not found in checkpoint. Optimizer reinitialized.")

            # Andere Zustände laden
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.memory = deque(checkpoint.get('memory', []), maxlen=self.memory.maxlen)

            print("Model and memory loaded successfully.")

