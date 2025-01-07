import os
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
from pyboy.utils import WindowEvent
import multiprocessing
import time

def evaluate_model(model_path, num_episodes=100, max_steps_per_episode=3000):
    """
    Evaluates a single trained model and shows its performance in the game.

    Args:
        model_path (str): Path to the trained model.
        num_episodes (int): Number of episodes to evaluate.
        max_steps_per_episode (int): Maximum steps per episode.
    """
    # Path to the ROM file
    rom_path = os.path.join("Game", "Kirby.gb")

    # Create the Kirby environment
    env = KirbyEnvironment(rom_path)
    print(f"Kirby environment successfully created for model: {model_path}")

    # Define the action mapping
    action_mapping = {
        0: [WindowEvent.PRESS_ARROW_RIGHT],
        1: [WindowEvent.PRESS_ARROW_LEFT],
        2: [WindowEvent.PRESS_BUTTON_A],
        3: [WindowEvent.PRESS_BUTTON_B],
        4: [WindowEvent.PRESS_ARROW_UP],
        5: [WindowEvent.PRESS_ARROW_DOWN],
        6: [
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
        ],
    }

    # Initialize the DDQN agent
    state_size = env.observation_space.shape[0]
    agent = DDQNAgent(state_size, len(action_mapping))

    # Load the trained model
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Model successfully loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}. Skipping...")
        env.pyboy.stop()
        return

    # Perform evaluation
    for episode in range(num_episodes):
        print(f"\nStarting evaluation episode {episode + 1}/{num_episodes} for model: {model_path}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Select action using the agent (exploit only)
            action_idx = agent.select_action(state)
            action = action_mapping[action_idx]

            # Perform the action
            for event in action:
                env.pyboy.send_input(event)

            # Retrieve the next state
            next_state, reward, done, info = env.step(action_idx)
            total_reward += reward
            state = next_state
            steps += 1

            # Check if the level is completed
            if info.get("level_complete"):
                print(f"Level completed in episode {episode + 1} for model: {model_path}")
                break

        print(f"Episode {episode + 1} for model {model_path} ended with total reward: {total_reward}")
        time.sleep(1)  # Pause between episodes
    print(f"Evaluation of model {model_path} completed. Closing environment.")
    env.pyboy.stop()

def evaluate_models_parallel(model_paths, num_episodes=5, max_steps_per_episode=3000):
    """
    Evaluates multiple trained models in parallel using multiprocessing.

    Args:
        model_paths (list): List of file paths for the trained models.
        num_episodes (int): Number of episodes to evaluate for each model.
        max_steps_per_episode (int): Maximum steps per episode.
    """
    processes = []
    for model_path in model_paths:
        # Start a separate process for each model
        process = multiprocessing.Process(
            target=evaluate_model,
            args=(model_path, num_episodes, max_steps_per_episode)
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

if __name__ == "__main__":
    # List of model paths
    model_paths = [
        "model/agent_model.pth",
        "model/agent_model_1000.pth",
        "model/agent_model_8000.pth",
        #"model/agent_model_25000.pth"
    ]

    # Start parallel evaluation
    evaluate_models_parallel(model_paths)
