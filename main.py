import os
import time
from Game.agents.agent import RandomAgent
from Game.game_env.environment import KirbyEnvironment
from pyboy.utils import WindowEvent

def main():
    # Specify the path to the ROM file
    rom_path = os.path.join("Game", "Kirby.gb")
    
    # Create the Kirby environment
    env = KirbyEnvironment(rom_path)
    print("Kirby environment successfully created! The game should now be running in the window.")

    # Define possible actions
    actions = [
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
    ]
    
    # Initialize the Random Agent
    agent = RandomAgent(actions)

    # Main game loop for the agent to take random actions
    try:
        while(True):  # Specify a limited number of steps for testing
            action = agent.select_action()
            #print (_)
            env.step(action)  # Use env.step(action) instead of directly calling send_input
            
            # Optional delay to control the frame rate
           # time.sleep()
    except KeyboardInterrupt:
        print("Game interrupted by the user.")
    
    # Cleanly close the environment
    env.close()
    print("PyBoy has been successfully closed.")

if __name__ == "__main__":
    main()