from pyboy import PyBoy
from pyboy.plugins.game_wrapper_kirby_dream_land import GameWrapperKirbyDreamLand

class KirbyEnvironment:
    def __init__(self, rom_path: str):
        # Initialize PyBoy with the Kirby ROM
        self.pyboy = PyBoy(rom_path, window="SDL2", sound=False)
        
        # Access the Kirby game wrapper directly without calling it
        self.kirby = self.pyboy.game_wrapper
        
        # Verify we have the correct game wrapper
        if not isinstance(self.kirby, GameWrapperKirbyDreamLand):
            raise TypeError("Loaded ROM does not correspond to Kirby's Dream Land.")
        
        # Start the game using the wrapper
        self.kirby.start_game()

        # Initialize previous state
        self.previous_score = self.kirby.score
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left

    def calculate_reward(self):
        reward = 0
        current_score = self.kirby.score
        current_health = self.kirby.health
        current_lives = self.kirby.lives_left

        # Reward for increased score
        if current_score > self.previous_score:
            reward += (current_score - self.previous_score)

        # Penalty for losing health
        if current_health < self.previous_health:
            reward -= (self.previous_health - current_health) * 10

        # Penalty for losing a life
        if current_lives < self.previous_lives:
            reward -= 50

        # Update previous state
        self.previous_score = current_score
        self.previous_health = current_health
        self.previous_lives = current_lives

        return reward

    def step(self, action):
        # Send the action to the game environment
        self.pyboy.send_input(action)
        self.pyboy.tick()

        # Calculate reward based on new state
        reward = self.calculate_reward()

        # Check if the game is over
        done = self.kirby.game_over()

        return reward, done

    def reset(self):
        # Reset the game environment to the initial state
        self.kirby.reset_game()
        self.previous_score = self.kirby.score
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left

    def get_game_area(self):
        # Get a 2D array representation of the game area
        return self.kirby.game_area()

    def close(self):
        self.pyboy.stop()
