import os
import time
from datetime import datetime
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
from pyboy.utils import WindowEvent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    # Erstelle einen eindeutigen Ordner für die Logs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/kirby_training_{current_time}"
    writer = SummaryWriter(log_dir)

    # Minimum number of steps to collect experiences before training starts
    BURN_IN_STEPS = 0

    # Specify the path to the ROM file
    rom_path = os.path.join("Game", "Kirby.gb")

    # Create the Kirby environment
    env = KirbyEnvironment(rom_path)
    print("Kirby environment successfully created! The game should now be running in the window.")

    # Define possible actions
    action_mapping = {
        0: [WindowEvent.PRESS_ARROW_RIGHT],
        1: [WindowEvent.PRESS_ARROW_LEFT],
        2: [WindowEvent.PRESS_BUTTON_A],
        3: [WindowEvent.PRESS_BUTTON_B],
        4: [WindowEvent.PRESS_ARROW_UP],
        5: [WindowEvent.PRESS_ARROW_DOWN],
        6: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],  # Right + Jump
        7: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],  # Right + Attack
        8: [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP],  # Jump + Up
        9: [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN]
    }

    # Initialize the DDQN Agent
    state_size = env.observation_space.shape[0]
    agent = DDQNAgent(state_size, len(action_mapping), memory_size=50000, batch_size=64)

    # Training settings
    num_epochs = 2000  # Gesamtzahl der Epochen
    max_steps_per_episode = 500

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    
    # Statistik-Tracking
    level_completions = 0    

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        level_completed = False

        with tqdm(total=max_steps_per_episode, desc=f"Epoch {epoch + 1} - Episode Progress", unit="step") as episode_progress:
            while not done and episode_length < max_steps_per_episode:
                # Wähle Aktion
                action_idx = agent.select_action(state)
                action = action_mapping[action_idx]

                # Führe die Aktion aus
                for event in action:
                    env.pyboy.send_input(event)

                # Schritt im Spiel
                next_state, reward, done, info = env.step(action_idx)
                total_reward += reward
                episode_length += 1

                # Fortschrittsbalken aktualisieren
                episode_progress.update(1)

                # Speichere Erfahrung
                agent.remember(state, action_idx, reward, next_state, done)

                # Aktualisiere Zustand
                state = next_state

                # Trainiere das Modell erst, wenn genügend Daten gesammelt wurden
                if len(agent.memory) >= BURN_IN_STEPS:
                    agent.train(epoch)

                # Wenn der Boss erreicht wurde, die Epoche beenden
                if info.get("level_complete"):  # Boss erreicht
                    level_completed = True
                    level_completions += 1
                    print("Boss erreicht! Epoche wird beendet und Level neu gestartet...")
                    done = True  # Epoche beenden

            print(f"\nEpisode ended. Reward: {total_reward}, Length: {episode_length}")

        # TensorBoard-Logs schreiben
        writer.add_scalar("Reward/Total", total_reward, epoch)
        writer.add_scalar("Episode Length", episode_length, epoch)
        writer.add_scalar("Epsilon", agent.epsilon, epoch)
        writer.add_scalar("Level Completions", level_completions, epoch)
        writer.add_scalar("Exploration/Epsilon", agent.epsilon, epoch)
        writer.add_scalar("Reward/Average", total_reward / max(1, episode_length), epoch)
        if level_completed:
            writer.add_scalar("Reward/Level Completed", total_reward, level_completions)        
        
        
        # Umgebung zurücksetzen
        state = env.reset()
        print(f"\nEpoch {epoch + 1} ended. Total Reward: {total_reward}")

        # Speichern des Modells nach jeder festgelegten Anzahl von Epochen
        if (epoch + 1) % 50 == 0:
            agent.save_model(checkpoint_path)
            print(f"Model and memory saved successfully to {checkpoint_path}.")
        
    # Speichere das endgültige Modell nach Abschluss des Trainings
    final_model_path = "agent_model.pth"
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}.")

    # Schließen des TensorBoard-Writers
    writer.close()

    print("Training complete.")

if __name__ == "__main__":
    main()