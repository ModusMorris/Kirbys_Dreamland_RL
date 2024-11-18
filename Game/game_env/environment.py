import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from PIL import Image
from pyboy.utils import WindowEvent


class KirbyEnvironment(gym.Env):
    def __init__(self, rom_path="Kirbys_Dream_Land.gb", action_mapping=None):
        super(KirbyEnvironment, self).__init__()
        self.pyboy = PyBoy(rom_path, window="SDL2", sound=False)
        self.kirby = self.pyboy.game_wrapper
        self.kirby.start_game()

        if action_mapping is None:
            raise ValueError("action_mapping cannot be None!")
        self.action_mapping = action_mapping

        self.action_space = gym.spaces.Discrete(len(action_mapping))
        self.observation_space = gym.spaces.Dict({
            "frames": gym.spaces.Box(low=0, high=255, shape=(4, 20, 16), dtype=np.float32),
            "game_area": gym.spaces.Box(low=0, high=255, shape=(18, 20), dtype=np.int32)
        })

        self.pyboy.set_emulation_speed(0)  # Max speed for training
        self.frame_stack = deque(maxlen=4)
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_score = self.kirby.score
        self.level_complete = False
        self.reset()

    def reset(self):
        """
        Setzt die Umgebung auf den Anfangszustand zurück.
        """
        self.kirby.reset_game()
        self.frame_stack.clear()
        initial_frame = self._get_screen_image()
        for _ in range(4):
            self.frame_stack.append(initial_frame)

        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_score = self.kirby.score
        self.level_complete = False

        return self._get_observation()

    def step(self, action, skip_frames=4):
        """
        Führt eine Aktion aus und berechnet die Belohnung sowie den nächsten Zustand.
        """
        for _ in range(skip_frames):
            self._perform_action(action)
            self.pyboy.tick()

        # Zustand aktualisieren
        new_frame = self._get_screen_image()
        self.frame_stack.append(new_frame)
        game_area = np.array(self.kirby.game_area(), dtype=np.int16)

        reward, done = self._calculate_reward()

        # Beobachtung erstellen
        observation = {
            "frames": np.stack(self.frame_stack, axis=0),
            "game_area": game_area
        }

        return observation, reward, done, {}

    def _get_screen_image(self):
        """
        Gibt das aktuelle Frame-Bild in Graustufen zurück.
        """
        screen_image = self.pyboy.screen.image.convert("L")
        resized_image = screen_image.resize((16, 20), Image.LANCZOS)
        screen_array = np.array(resized_image, dtype=np.float32) / 255.0
        return screen_array

    def _perform_action(self, action):
        """
        Führt eine Aktion aus, basierend auf dem Action-Mapping.
        """
        if action not in self.action_mapping:
            raise KeyError(f"Invalid action index: {action}")
        for event in self.action_mapping[action]:
            self.pyboy.send_input(event)

    def _get_kirby_position(self):
        """
        Gibt die x- und y-Position von Kirby zurück, basierend auf dem Speicher.
        """
        kirby_x = self.pyboy.memory[0xD05C]
        kirby_y = self.pyboy.memory[0xD05D]
        return kirby_x, kirby_y

    def _calculate_reward(self):
        """
        Berechnet die Belohnung basierend auf der aktuellen Umgebung und Kirbys Aktionen.
        """
        kirby_position = self._get_kirby_position()
        game_area = np.array(self.kirby.game_area(), dtype=np.int16)

        reward = 0.0
        done = False

        # Gesundheit prüfen
        current_health = self.kirby.health
        if current_health < self.previous_health:
            reward -= 10.0
            print("Kirby hat Gesundheit verloren!")
        self.previous_health = current_health

        # Leben prüfen
        current_lives = self.kirby.lives_left
        if current_lives < self.previous_lives:
            reward -= 50.0
            print("Kirby hat ein Leben verloren. Epoche wird neugestartet.")
            done = True  # Neustart der Epoche
        self.previous_lives = current_lives

        # Warpstar auf der Karte finden
        warpstar_position = self._find_warpstar_position(game_area)

        # Warpstar erreichen
        if warpstar_position != (-1, -1) and abs(kirby_position[0] - warpstar_position[0]) <= 1:
            reward += 100.0  # Große Belohnung für Warpstar
            print("Kirby hat den Warpstar erreicht!")
            done = True  # Beendet die Epoche

        # Punktzahl prüfen
        current_score = self.kirby.score
        if current_score > self.previous_score:
            reward += (current_score - self.previous_score) * 5.0
        self.previous_score = current_score

        return reward, done

    def _find_warpstar_position(self, game_area):
        """
        Sucht nach der Position des Warpstars in der Game Area.
        """
        warpstar_tile = 114  # Beispiel: Warpstar-Tile-ID
        for y, row in enumerate(game_area):
            for x, tile in enumerate(row):
                if tile == warpstar_tile:
                    return x, y
        return -1, -1

    def _get_observation(self):
        """
        Erstellt die Beobachtung bestehend aus Frames und der Game Area.
        """
        game_area = np.array(self.kirby.game_area(), dtype=np.int16)
        return {
            "frames": np.stack(self.frame_stack, axis=0),
            "game_area": game_area
        }
