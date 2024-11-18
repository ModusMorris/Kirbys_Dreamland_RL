import os
import time
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
from pyboy.utils import WindowEvent


def train_process(rank, model_path, frame_shape, game_area_shape, action_size, action_mapping, log_dir, num_epochs, max_steps_per_episode):
    writer = SummaryWriter(log_dir=f"{log_dir}_worker_{rank}")
    env = KirbyEnvironment("Game/Kirby.gb", action_mapping)
    agent = DDQNAgent(frame_shape, game_area_shape, action_size, model_path=model_path)
    burn_in_steps = 2000

    total_steps = 0
    with tqdm(total=num_epochs, desc=f"Worker {rank} - Training Epochs") as epoch_bar:
        for epoch in range(num_epochs):
            state = env.reset()
            total_reward = 0
            steps = 0
            start_time = time.time()
            done = False

            with tqdm(total=max_steps_per_episode, desc=f"Worker {rank} - Epoch {epoch + 1} Steps", leave=False) as step_bar:
                while not done and steps < max_steps_per_episode:
                    # Aktion auswählen
                    action_idx = agent.select_action(state)
                    next_state, reward, done, info = env.step(action_idx)

                    # Wenn ein Leben verloren wurde, breche die Episode ab
                    if info.get("life_lost", False):
                        print(f"Worker {rank} | Epoch {epoch + 1} | Kirby hat ein Leben verloren.")
                        done = True  # Beende die Episode

                    agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    total_steps += 1

                    # Training alle 10 Schritte nach Burn-in
                    if steps % 10 == 0 and len(agent.memory) > burn_in_steps:
                        agent.train(epoch)

                    # Update des Fortschritts
                    step_bar.update(1)

                # Epochendetails loggen
                elapsed_time = time.time() - start_time
                writer.add_scalar("Reward/Total", total_reward, epoch)
                writer.add_scalar("Steps/Episode Length", steps, epoch)
                writer.add_scalar("Time/Epoch Duration", elapsed_time, epoch)

                print(f"Worker {rank} | Epoch {epoch + 1}/{num_epochs} | Reward: {total_reward:.2f} | Steps: {steps} | Time: {elapsed_time:.2f}s")

                epoch_bar.update(1)

                # Epoche neu starten, wenn Leben verloren wurde
                if info.get("life_lost", False):
                    break

    # Modell und Logs speichern
    agent.save_model(f"{model_path}_worker_{rank}")
    writer.close()


def main():
    model_path = "agent_model.pth"
    log_dir = "runs/kirby_training"
    frame_shape = (4, 20, 16)
    game_area_shape = (16, 20)
    action_size = 10
    action_mapping = {
        0: [WindowEvent.PRESS_ARROW_RIGHT],
        1: [WindowEvent.PRESS_ARROW_LEFT],
        2: [WindowEvent.PRESS_BUTTON_A],
        3: [WindowEvent.PRESS_BUTTON_B],
        4: [WindowEvent.PRESS_ARROW_UP],
        5: [WindowEvent.PRESS_ARROW_DOWN],
        6: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
        7: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
        8: [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP],
        9: [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN],
    }
    num_epochs = 10
    max_steps_per_episode = 2000
    num_workers = 1

    # Multiprocessing
    mp.set_start_method("spawn")
    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=train_process, args=(
            rank, model_path, frame_shape, game_area_shape, action_size, action_mapping, log_dir, num_epochs, max_steps_per_episode))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
