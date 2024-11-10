import os
import time
from Game.game_env.environment import KirbyEnvironment
from Game.agents.agent import DDQNAgent
import torch
from pyboy.utils import WindowEvent

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
    num_episodes = 100  # Festgelegte Anzahl an Episoden
    
    # Initialize the DDQN Agent
    agent = DDQNAgent(state_size, len(action_mapping))

    for episode in range(num_episodes):
        state = env.reset()
        state = env.get_state()
        done = False
        total_reward = 0

        while not done:
            action_idx = agent.select_action(state)
            action = action_mapping[action_idx]
            
            # Execute action
            for event in action:
                env.pyboy.send_input(event)
            
            # Step in environment
            reward, done = env.step(action_idx)
            next_state = env.get_state()
            total_reward += reward

            # Store experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state
            state = next_state

            if done:
                print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

    env.close()
    print("PyBoy has been successfully closed.")

if __name__ == "__main__":
    main()