# 🎮 Reinforcement Learning: Playing Kirby's Dreamland 🌟

![Kirby complete level](https://github.com/user-attachments/assets/73091eef-1614-465c-a7f1-7f92a33a9209)



## 📝 Project Overview
Welcome to the Kirby's Dreamland RL Project! This project showcases the implementation of a Reinforcement Learning (RL) agent that learns to play the first level of Kirby's Dreamland. Using a Double Deep Q-Network (DDQN), the agent aims to:
- 🌠 Complete the level by reaching the warp star.
- 💀 Avoid losing all lives (Game Over).
  
Key features of the project include:
- 🖥️ Efficient training using a Convolutional Neural Network (CNN) that processes game data as arrays, not raw images.
- 🏆 A custom reward and penalty system to guide the agent’s progress.
## 🔑 Key Details
### 🌍 Environment
This project uses PyBoy, a Game Boy emulator, as the training environment. Game states are extracted as arrays representing the game field instead of relying on raw image data.
🎮 Action Space: Kirby can perform various actions such as:
- Moving left or right
- Jumping
- flying
- absorb in enemies and items
- Attacking enemies
- 🖼️ Observation Space: The game area is represented as a continuously updated 2D array.
## 🧠 Training Details
The RL agent was trained for 25000 epochs, with each episode defined as:
- 🎯 Completion: Kirby reaches the warp star.
- 💀 Termination: Kirby loses all lives.
- ⏱️ Timeout: Kirby takes 2500 steps without completing the level.
### 📊 Rewards and Penalties
The agent is guided by a custom reward system:
- 🥇 Rewards:
  - Progressing toward the goal (e.g., moving right).
  - Defeating enemies or completing the level.
- ⚠️ Penalties:
  - Standing idle or moving away from the goal
  - Losing health or a life.
### 🧩 Neural Network Architecture
- 🤖 Model: Double Deep Q-Network (DDQN).
- 🌀 Feature Extractor: CNN layers to process game field arrays.
- ⚡ Efficiency: By training on arrays instead of raw images, the agent achieves:
  - 🚀 Faster training.
  - 🛠️ Reduced resource usage.
  - 📈 Maintained accuracy.

 
## ⚡ Why This Approach?
Training directly on raw game images is computationally expensive and often unnecessary. By representing the game state as numerical arrays:
- 🛠️ Resources: Significantly reduced computational requirements.
- 🚀 Speed: Faster training with efficient memory usage.
- 🎯 Focus: Enables the agent to learn the most relevant patterns in the game environment.

## 🚀 Getting Started
### 🔧 Setup
1. Clone the repository:
  `git clone https://github.com/your-repo/kirby-rl.git`
  `cd kirby-rl`
2. Install the dependencies:
   `pip install -r requirements.txt`
3. Place the Kirby.gb ROM file in the appropriate directory.

### 🏋️ Run Training
Start training the agent by running:
`python main.py`
### 🔄 Resume Training
To resume training from the last checkpoint:
- The model will automatically load the latest saved state.
- Exploration (epsilon) will adjust dynamically to continue learning effectively.

## 🌟 Results
After training for 25000 epochs:
- 🎉 The agent completes the first level by reaching the warp star.
- 📈 Learning is guided by a reward system that encourages progress and discourages inefficient behavior.

## 🌍 Future Work
💡 Potential enhancements for the project:
1. 🕹️ Extend Training: Apply the RL agent to additional levels of the game.
2. 📊 Compare Algorithms: Experiment with alternative RL models for better performance.
3. 🔧 Reward Tuning: Refine the reward system for more complex scenarios.
