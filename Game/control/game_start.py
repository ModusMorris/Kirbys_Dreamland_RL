import time
from pyboy.utils import WindowEvent

def start_game(pyboy):
    """
    Starts the game by pressing the START button twice.
    
    Args:
    - pyboy: The PyBoy instance
    """
    print("Press START to begin the game...")
    for _ in range(2):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_START)  # Press START
        pyboy.tick()  # One step in the emulator
        time.sleep(0.5)  # Short wait between "Start" button presses
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)  # Release START
        pyboy.tick()  # One step in the emulator
        time.sleep(0.5)  # Wait for the game to react
    print("The level has begun!")
