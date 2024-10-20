from pyboy import PyBoy

def create_kirby_environment(rom_path: str):
    """
    Creates a PyBoy environment for Kirby's Dreamland and starts the game.
    
    Args:
    - rom_path: Path to the Kirby's Dreamland ROM file.
    
    Returns:
    - pyboy: The running PyBoy instance.
    """
    try:
        # Initialize PyBoy with window display (SDL2)
        pyboy = PyBoy(rom_path, window_type="SDL2", sound=False)
        if pyboy:
            print("PyBoy was successfully created.")
        return pyboy
    except Exception as e:
        print(f"Error creating the PyBoy environment: {e}")
        return None
