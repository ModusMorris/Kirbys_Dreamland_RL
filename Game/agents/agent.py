import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import os

# Disable certain TensorFlow optimizations for compatibility (though not directly used here)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class DDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size=256,
        gamma=0.97,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999999975,
        target_update_frequency=5000,
        memory_size=500000,
        model_path="agent_model.pth",
        writer=None,
        **kwargs,
    ):
        """
        Initializes the Double Deep Q-Network (DDQN) agent.

        Args:
            state_size (int): Dimension of the input state space.
            action_size (int): Number of possible actions.
            batch_size (int): Size of batches used for training.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizer.
            epsilon_start (float): Initial exploration probability.
            epsilon_end (float): Minimum exploration probability.
            epsilon_decay (float): Decay rate for epsilon over time.
            target_update_frequency (int): Number of steps before updating the target network.
            memory_size (int): Maximum size of the replay memory.
            model_path (str): Path for saving and loading the model.
            writer (SummaryWriter): TensorBoard writer for logging training metrics.
        """
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

        # Use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the main and target networks on the chosen device
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()  # Synchronize the target model with the main model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_model()  # Load the model from file if it exists

    def _build_model(self):
        """
        Constructs the neural network used for predicting Q-values.
        The architecture includes convolutional layers for feature extraction
        followed by fully connected layers for decision making.
        """
        model = nn.Sequential(
            nn.Conv2d(
                4, 32, kernel_size=3, stride=1, padding=1
            ),  # Input: 4 frames, Output: 32 feature maps
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # Downsamples by a factor of 2
            nn.ReLU(),
            nn.Flatten(),  # Flattens the output for the fully connected layers
            nn.Linear(64 * 10 * 8, 128),  # Adjusted to match flattened size
            nn.ReLU(),
            nn.Linear(
                128, self.action_size
            ),  # Output layer with Q-values for each action
        )
        return model

    def update_target_model(self):
        """
        Synchronizes the target network with the main network.
        This is used in Double Q-Learning to stabilize training.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory for later training.
        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state after the action.
            done (bool): Whether the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        Explores with probability epsilon and exploits otherwise.

        Args:
            state (np.array): Current state.

        Returns:
            int: Selected action index.
        """
        if random.random() <= self.epsilon:  # Exploration
            return random.randrange(self.action_size)
        else:  # Exploitation
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def replay(self, current_epoch):
        """
        Trains the model using a batch of experiences from the replay memory.

        Args:
            current_epoch (int): Current training epoch, used for logging.
        """
        if len(self.memory) < self.batch_size:  # Skip training if not enough samples
            return

        # Sample a random batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert data to PyTorch tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(list(actions)).to(self.device)
        rewards = torch.FloatTensor(list(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(list(dones)).to(self.device)

        # Compute current Q-values and target Q-values for Double Q-Learning
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Calculate the loss and backpropagate
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss to TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss", loss.item(), current_epoch)

        # Gradually decrease epsilon for exploration-exploitation tradeoff
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def train(self, current_epoch):
        """
        Performs multiple replay steps to train the network.
        Also updates the target model periodically.

        Args:
            current_epoch (int): Current training epoch, used for logging.
        """
        for _ in range(5):  # Train on smaller batches
            self.replay(current_epoch)

        # Periodically update the target network
        if len(self.memory) % self.target_update_frequency == 0:
            self.update_target_model()

    def save_model(self, checkpoint_path=None):
        """
        Saves the model, optimizer state, epsilon value, and replay memory to a file.

        Args:
            checkpoint_path (str): Path to save the checkpoint. Defaults to self.model_path.
        """
        if checkpoint_path is None:
            checkpoint_path = self.model_path

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "memory": list(self.memory),
            },
            checkpoint_path,
        )
        print(f"Model weights saved successfully to {checkpoint_path}.")

    def load_model(self, model_path=None):
        """
        Loads the model, optimizer state, epsilon value, and replay memory from a file.

        Args:
            model_path (str): Path to the model file to load. Defaults to self.model_path.
        """
        if model_path is None:
            model_path = self.model_path

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(
                checkpoint.get("optimizer_state_dict", self.optimizer.state_dict())
            )
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.memory = deque(checkpoint.get("memory", []), maxlen=self.memory.maxlen)
            print(f"Model and memory loaded successfully from {model_path}.")
        else:
            print(f"Model file not found: {model_path}")
