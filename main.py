import os
import time
import signal  # New: For signal handling
from datetime import datetime
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
from pyboy.utils import WindowEvent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Global variable for the agent
agent = None


def handle_exit_signal(signum, frame):
    """
    Handles exit signals (e.g., SIGINT, SIGTERM) to ensure the model is saved before exiting.
    """
    print("\nExit signal received. Saving model...")
    if agent is not None:
        final_model_path = "agent_model.pth"
        agent.save_model(final_model_path)
        print(f"Model successfully saved at {final_model_path}.")
    exit(0)


def main():
    global agent  # Make the agent accessible to the signal handler

    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    # Create a unique folder for logs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/kirby_training_{current_time}"
    writer = SummaryWriter(log_dir)

    # Specify the path to the ROM file
    rom_path = os.path.join("Game", "Kirby.gb")

    # Create the Kirby game environment
    env = KirbyEnvironment(rom_path)
    print(
        "Kirby environment successfully created! The game should now be running in the window."
    )

    # Define possible actions
    action_mapping = {
        0: [WindowEvent.PRESS_ARROW_RIGHT],
        1: [WindowEvent.PRESS_ARROW_LEFT],
        2: [WindowEvent.PRESS_BUTTON_A],
        3: [WindowEvent.PRESS_BUTTON_B],
        4: [WindowEvent.PRESS_ARROW_UP],
        5: [WindowEvent.PRESS_ARROW_DOWN],
        # 6: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],  # Right + Jump
        # 7: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],  # Right + Attack
        # 8: [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP],  # Jump + Up
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
    agent = DDQNAgent(state_size, len(action_mapping), memory_size=50000, batch_size=64)

    # Training settings
    num_epochs = 8000  # Total number of epochs
    max_steps_per_episode = 3000

    # Create a directory for checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

    # Track statistics
    level_completions = 0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        level_completed = False

        with tqdm(
            total=max_steps_per_episode,
            desc=f"Epoch {epoch + 1} - Episode Progress",
            unit="step",
        ) as episode_progress:
            while not done and episode_length < max_steps_per_episode:
                # Select an action
                action_idx = agent.select_action(state)
                action = action_mapping[action_idx]

                # Perform the action
                for event in action:
                    env.pyboy.send_input(event)

                # Step through the game
                next_state, reward, done, info = env.step(action_idx)
                total_reward += reward
                episode_length += 1

                # Update progress bar
                episode_progress.update(1)

                # Store the experience
                agent.remember(state, action_idx, reward, next_state, done)

                # Update the current state
                state = next_state

                # Train the agent
                agent.train(epoch)

                # End the epoch if level is completed or Kirby loses a life
                if info.get("level_complete"):  # Level completed
                    level_completed = True
                    level_completions += 1
                    print("Level completed, ending epoch.")
                    done = True
                elif info.get("life_lost"):  # Kirby loses a life
                    print("Kirby lost a life! Ending epoch.")
                    done = True

            print(f"\nEpisode ended. Reward: {total_reward}, Length: {episode_length}")

        # Log statistics to TensorBoard
        writer.add_scalar("Reward/Total", total_reward, epoch)
        writer.add_scalar("Episode Length", episode_length, epoch)
        writer.add_scalar("Epsilon", agent.epsilon, epoch)
        writer.add_scalar("Level Completions", level_completions, epoch)
        writer.add_scalar("Exploration/Epsilon", agent.epsilon, epoch)
        writer.add_scalar(
            "Reward/Average", total_reward / max(1, episode_length), epoch
        )
        if level_completed:
            writer.add_scalar("Reward/Level Completed", total_reward, level_completions)

        print(f"\nEpoch {epoch + 1} ended. Total Reward: {total_reward}")

        # Reset environment
        state = env.reset()
        total_reward = 0  # Ensure total_reward is reset

        # Save the model after every set number of epochs
        if (epoch + 1) % 50 == 0:
            agent.save_model(checkpoint_path)
            print(f"Model and memory saved successfully to {checkpoint_path}.")

    # Save the final model after training completes
    final_model_path = "agent_model.pth"
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}.")

    # Close the TensorBoard writer
    writer.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
