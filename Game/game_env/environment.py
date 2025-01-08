import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from pyboy.utils import WindowEvent


class KirbyEnvironment(gym.Env):
    def __init__(self, rom_path="Kirby.gb"):
        super(KirbyEnvironment, self).__init__()
        self.pyboy = PyBoy(rom_path, window="SDL2")
        self.pyboy.set_emulation_speed(2)
        self.kirby = self.pyboy.game_wrapper

        # Start the game
        self.kirby.start_game()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, *self.pyboy.game_area().shape), dtype=np.uint8
        )

        # Frame stack for 4 frames
        self.frame_stack = deque(maxlen=4)w

        # Initialize state information
        self.previous_health = None
        self.previous_lives = None
        self.previous_score = None
        self.previous_position = (0, 0)
        self.previous_boss_health = None
        self.previous_level_progress = None
        self.previous_game_state = 1
        self.y_axis_steps = 0  # New: Count of steps only on the y-axis

    def IsBossActive(self):
        # Check if the boss is active
        current_boss_health = self.pyboy.memory[0xD093]
        return current_boss_health > 0

    def reset(self):
        self.kirby.reset_game()
        self.frame_stack.clear()

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
        kirby_x_position = self.pyboy.memory[0xD05C]
        reward = 0
        level_complete = False
        life_lost = False

        # 1. Bestrafung für Verlust von Leben beim Besiegen von Gegnern
        if current_lives < self.previous_lives:
            print("Kirby lost a life!")
            reward -= 10000  # Hohe Strafe für Verlust von Leben
            life_lost = True

        # 2. Belohnung für Besiegen eines Gegners ohne Schaden
        if current_score > self.previous_score and current_health == self.previous_health:
            print("Enemy defeated without taking damage!")
            reward += 500  # Zusätzliche Belohnung für sauberes Besiegen von Gegnern

        # 3. Bestrafung für Schaden beim Besiegen eines Gegners
        if current_score > self.previous_score and current_health < self.previous_health:
            print("Enemy defeated, but Kirby took damage!")
            reward -= 200  # Bestrafung, wenn Kirby Schaden nimmt

        # 4. Boss besiegt
        if current_boss_health == 0 and self.previous_boss_health > 0:
            print("Boss defeated!")
            reward += 15000  # Hohe Belohnung für Besiegen des Bosses

        # 5. Boss beschädigt
        if current_boss_health < self.previous_boss_health:
            print("Boss damaged!")
            reward += 2000  # Moderate Belohnung für Schaden am Boss

        # 6. Fortschritt (Bewegung nach rechts)
        if current_level_progress > self.previous_level_progress:
            reward += 15  # Kleine Belohnung für Fortschritt

        # 7. Rückschritt (Bewegung nach links)
        if current_level_progress < self.previous_level_progress:
            reward -= 10  # Kleine Strafe für Rückschritt

        # 8. Warpstar erreicht
        if (
            current_health > 0
            and current_game_state == 6
            and self.previous_game_state != 6
        ):
            print("Warpstar reached!")
            level_complete = True
            reward += 25000  # Große Belohnung für Abschluss des Levels

        # 9. Stillstand bestrafen
        if current_level_progress == self.previous_level_progress:
            reward -= 2  # Strafe für keinen Fortschritt

        # 10. Belohnung für effizientes Spielen
        if current_health == self.previous_health and not life_lost:
            reward += 5  # Belohnung für keine Verluste während des Fortschritts

        # Update States
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
