import os
import time
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
import torch
from pyboy.utils import WindowEvent
from tqdm import tqdm

def main():
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
        6: [
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN
        ]
    }
    
    state_size = 5  # Anzahl der Elemente im Zustand
    
    # Initialize the DDQN Agent with a Replay Memory
    agent = DDQNAgent(state_size, len(action_mapping))
    
    # Training für 100 Epochen
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Wähle Aktion
            action_idx = agent.select_action(state)
            action = action_mapping[action_idx]
            
            # Führe die Aktion aus
            for event in action:
                env.pyboy.send_input(event)
            
            # Schritt im Spiel
            reward, done = env.step(action_idx)
            next_state = env.get_state()
            total_reward += reward

            # Speichere Erfahrung
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Aktualisiere Zustand
            state = next_state

            # Trainiere das Modell nach jeder Aktion
            agent.train()

            if done:
                print(f"Epoch {epoch + 1} ended. Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

    # Speichere das Modell nach dem Training
    agent.save_model()  
    print("Training complete. Model saved.")

    # Evaluierung des trainierten Modells
    print("\nStarting evaluation...")
    agent.epsilon = 0  # Setze epsilon auf 0, um nur das gelernte Verhalten zu nutzen
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Wähle Aktion nur basierend auf dem gelernten Modell
        action_idx = agent.select_action(state)
        action = action_mapping[action_idx]

        # Führe die Aktion aus
        for event in action:
            env.pyboy.send_input(event)
        
        # Schritt im Spiel
        reward, done = env.step(action_idx)
        next_state = env.get_state()
        total_reward += reward
        
        # Aktualisiere Zustand
        state = next_state

        if done:
            print(f"Evaluation ended. Total Reward: {total_reward}")
            break

    env.close()
    print("PyBoy has been successfully closed.")

if __name__ == "__main__":
    main()
