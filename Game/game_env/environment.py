import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from pyboy.utils import WindowEvent


class KirbyEnvironment(gym.Env):
    def __init__(self, rom_path="Kirby.gb"):
        super(KirbyEnvironment, self).__init__()
        self.pyboy = PyBoy(rom_path, window="SDL2")
        self.pyboy.set_emulation_speed(0)
        self.kirby = self.pyboy.game_wrapper

        # Start the game
        self.kirby.start_game()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, *self.pyboy.game_area().shape), dtype=np.uint8
        )

        # Frame stack for 4 frames
        self.frame_stack = deque(maxlen=4)

        # Initialize state information
        self.previous_health = None
        self.previous_lives = None
        self.previous_score = None
        self.previous_position = (0, 0)
        self.previous_boss_health = None
        self.previous_level_progress = None
        self.previous_game_state = 1
        self.y_axis_steps = 0  # New: Count of steps only on the y-axis
        self.step_count = 0  # New: Track the number of steps in the episode

    def IsBossActive(self):
        # Check if the boss is active
        current_boss_health = self.pyboy.memory[0xD093]
        return current_boss_health > 0

    def reset(self):
        self.kirby.reset_game()
        self.frame_stack.clear()
        self.step_count = 0  # Reset step count

        # Initialize the frame stack with 4 identical frames
        initial_observation = self.pyboy.game_area()
        for _ in range(4):
            self.frame_stack.append(initial_observation)

        # Initialize state information
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_score = self.kirby.score
        self.previous_position = (self.pyboy.memory[0xD05C], self.pyboy.memory[0xD05D])
        self.previous_boss_health = self.pyboy.memory[0xD093]
        self.previous_level_progress = self._calculate_level_progress()
        self.previous_game_state = self.pyboy.memory[0xD02C] = 1
        self.y_axis_steps = 0  # Reset y-axis steps

        return np.stack(self.frame_stack, axis=0)

    def step(self, action):
        self._perform_action(action)
        self.pyboy.tick()
        self.step_count += 1  # Increment step count

        # Update the frame stack
        self.frame_stack.append(self.pyboy.game_area())

        # Create the observation (4 stacked frames)
        next_state = np.stack(self.frame_stack, axis=0)

        # Calculate the reward and check if the level is completed
        reward, level_complete, life_lost = self._calculate_reward()

        # Handle boss completion and life loss
        if level_complete:
            return next_state, reward, True, {"level_complete": True}

        if life_lost:
            return (
                next_state,
                reward,
                True,
                {"life_lost": True, "level_complete": False},
            )

        # Default case: Continue training
        done = self._check_done()  # Check if Kirby loses all lives or the game is over
        return next_state, reward, done, {"level_complete": False, "life_lost": False}

    def _get_observation(self):
        # Retrieve game area as observation
        return self.pyboy.game_area()

    def _perform_action(self, action):
        actions = [
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
        ]
        if action < len(actions):
            self.pyboy.send_input(actions[action])
        self.pyboy.tick()

    def _calculate_level_progress(self):
        screen_x_position = self.pyboy.memory[0xD053]
        kirby_x_position = self.pyboy.memory[0xD05C]
        scx = screen_x_position * 16 + kirby_x_position
        return scx

    def _calculate_reward(self):
        current_health = self.kirby.health
        current_lives = self.kirby.lives_left
        current_score = self.kirby.score
        current_boss_health = self.pyboy.memory[0xD093]
        current_game_state = self.pyboy.memory[0xD02C]
        current_level_progress = self._calculate_level_progress()
        current_position = (self.pyboy.memory[0xD05C], self.pyboy.memory[0xD05D])
        reward = 0
        level_complete = False
        life_lost = False

        # 1. Progression Reward with Time Factor
        progress_reward = max(0, current_level_progress - self.previous_level_progress)
        time_factor = max(0.5, 1 - (self.step_count / 2500))  # Minimum 50% reward scaling
        reward += progress_reward * time_factor

        # 2. Boss defeated
        if current_boss_health == 0 and self.previous_boss_health > 0:
            print("Boss defeated")
            reward += 10000

        # 3. Damage dealt to the boss
        if current_boss_health < self.previous_boss_health:
            print("Boss damaged")
            reward += 1000

        # 4. Loss of a life
        if current_lives < self.previous_lives:
            reward -= 3000
            self.kirby.reset_game()
            life_lost = True

        # 5. Loss of health
        if current_health < self.previous_health:
            reward -= (self.previous_health - current_health) * 50

        # 6. Moving right
        if current_position[0] > self.previous_position[0]:
            reward += 10

        # 7. Kirby stands still
        if current_level_progress == self.previous_level_progress:
            reward -= 1  # Reduce penalty to encourage exploration

        # 8. Kirby only moves on the y-axis
        if current_position[0] == self.previous_position[0]:
            if current_position[1] != self.previous_position[1]:
                self.y_axis_steps += 1
            else:
                self.y_axis_steps = 0

            if self.y_axis_steps > 200:
                reward -= 40
                self.y_axis_steps = 0
        else:
            self.y_axis_steps = 0

        # 9. Score increase
        if current_score > self.previous_score:
            reward += (current_score - self.previous_score) * 0.1

        # 10. Warpstar reached
        if (
            current_health > 0
            and current_game_state == 6
            and self.previous_game_state != 6
        ):
            print(current_game_state)
            print("Warpstar reached")
            level_complete = True
            reward += 20000

        # 11. Time Penalty with Maximum Limit
        max_time_penalty = 500  # Maximum penalty cap
        reward -= min(self.step_count * 0.01, max_time_penalty)

        # Log rewards for debugging
        #print(f"Step Count: {self.step_count}, Time Penalty: {min(self.step_count * 0.01, max_time_penalty)}, Progress Reward: {progress_reward}")

        # Update states
        self.previous_health = current_health
        self.previous_lives = current_lives
        self.previous_score = current_score
        self.previous_boss_health = current_boss_health
        self.previous_level_progress = current_level_progress
        self.previous_game_state = current_game_state
        self.previous_position = current_position

        return reward, level_complete, life_lost

    def _check_done(self):
        return self.kirby.game_over()
