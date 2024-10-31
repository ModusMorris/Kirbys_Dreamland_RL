import time
from pyboy.utils import WindowEvent

def random_agent_actions(pyboy, agent, buttons):
    """
    Executes random actions through the agent.
    
    Args:
    - pyboy: The PyBoy instance
    - agent: The RandomAgent instance
    - buttons: List of buttons that can be pressed
    """
    try:
        # Loop to let the agent perform random actions
        while True:
            # Select a random action
            action = agent.select_action()

            # Press the button in the emulator
            pyboy.send_input(action)

            # Perform a tick in the emulator (advance the game)
            pyboy.tick()

            # Optional delay to control the frame rate
            time.sleep(0.05)

            # Release the button to ensure no button remains pressed
            if action == WindowEvent.PRESS_ARROW_UP:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
            elif action == WindowEvent.PRESS_ARROW_DOWN:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            elif action == WindowEvent.PRESS_ARROW_LEFT:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            elif action == WindowEvent.PRESS_ARROW_RIGHT:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            elif action == WindowEvent.PRESS_BUTTON_A:
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            elif action == WindowEvent.PRESS_BUTTON_B:
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)

            # An additional tick to ensure the emulator processes the action
            pyboy.tick()

    except KeyboardInterrupt:
        # Safely exit the game if the user presses CTRL+C
        print("Game is being terminated by the user...")
