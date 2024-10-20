import os
import time
from Game.agents.agent import RandomAgent
from Game.game_env.environment import create_kirby_environment
from Game.control.game_start import start_game
from Game.control.agent_actions import random_agent_actions
from pyboy.utils import WindowEvent

def main():
    # Specify the path to the ROM file (absolute or relative)
    rom_path = os.path.join("Game", "Kirby.gb")
    
    # Create the emulator environment
    pyboy = create_kirby_environment(rom_path)
    if not pyboy:
        print("Error: The Kirby environment could not be created.")
        return

    print("Kirby environment successfully created! The game should now be running in the window.")

    # Set the emulator to turbo mode
    pyboy.set_emulation_speed(0)  # Sets the speed to maximum

    # Define the possible buttons that can be pressed
    buttons = [
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
    ]
    
    # Initialize the Random Agent
    agent = RandomAgent(buttons)

    # Give the window time to become visible
    time.sleep(3)

    # Phase 1: Start the game
    start_game(pyboy)

    # Phase 2: Execute random actions
    random_agent_actions(pyboy, agent, buttons)

    # Cleanly close PyBoy
    pyboy.stop()
    print("PyBoy has been successfully closed.")


if __name__ == "__main__":
    main()
