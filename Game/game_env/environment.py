import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from pyboy.utils import WindowEvent

class KirbyEnvironment(gym.Env):
    def __init__(self, rom_path="Kirby.gb"):
        super(KirbyEnvironment, self).__init__()
        self.pyboy = PyBoy(rom_path, window="null")
        self.pyboy.set_emulation_speed(0)
        self.kirby = self.pyboy.game_wrapper

        # Start des Spiels
        self.kirby.start_game()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, *self.pyboy.game_area().shape), dtype=np.uint8
        )
        
        # Frame stack for 4 frames
        self.frame_stack = deque(maxlen=4)

        # Initialisiere Zustandsinformationen
        self.previous_health = None
        self.previous_lives = None
        self.previous_score = None
        self.previous_position = (0, 0)

    def reset(self):
        if not self.pyboy.tick():
            self.kirby.start_game()

        self.frame_stack.clear()

        # Initialisiere den Frame-Stack mit 4 identischen Frames
        initial_observation = self.pyboy.game_area()
        for _ in range(4):
            self.frame_stack.append(initial_observation)

        # Initialisieren der Zustandsinformationen
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_score = self.kirby.score
        self.previous_position = (self.pyboy.memory[0xD05C], self.pyboy.memory[0xD05D])

        return np.stack(self.frame_stack, axis=0)

    def step(self, action):
        self._perform_action(action)
        self.pyboy.tick()

        # Update the frame stack
        self.frame_stack.append(self.pyboy.game_area())

        # Erstelle die Beobachtung (4 gestapelte Frames)
        next_state = np.stack(self.frame_stack, axis=0)

        # Belohnung berechnen und überprüfen, ob das Level abgeschlossen ist
        reward, level_complete = self._calculate_reward()

        # Behandle Boss-Erreichen und Lebensverlust
        if level_complete:
            return next_state, reward, True, {"level_complete": True}

        # Standard-Fall: Weitertrainieren
        done = self._check_done()  # Kirby verliert alle Leben oder das Spiel ist vorbei
        return next_state, reward, done, {"level_complete": False}
    
    # def _find_star_in_game_area(self):
    #     game_area = self.pyboy.game_area()
        
    #     # Pixelwert des Sterns (dieser Wert muss angepasst werden)
    #     STAR_PIXEL_VALUE = 200  # Beispiel: Pixelwert, der den Stern repräsentiert
        
    #     # Durchsuche das gesamte `game_area`-Array nach dem Pixelwert
    #     for y, row in enumerate(game_area):
    #         for x, pixel in enumerate(row):
    #             if pixel == STAR_PIXEL_VALUE:  # Stern erkannt
    #                 return (x, y)  # Position des Sterns
    #     return None  # Kein Stern gefunden'
    
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

    def _calculate_reward(self):
        kirby_x = self.pyboy.memory[0xD05C]
        kirby_y = self.pyboy.memory[0xD05D]
        boss_health = self.pyboy.memory[0xD093]  # Gesundheitsstatus des Bosses
        current_health = self.kirby.health
        current_lives = self.kirby.lives_left
        current_score = self.kirby.score
        reward = 0
        level_complete = False

        # Belohnung für Bewegung nach rechts (Fortschritt im Level)
        if kirby_x > self.previous_position[0]:
            reward += 1  # Fortschrittsbelohnung
            
        # Bestrafung für Bewegung nach links
        if kirby_x < self.previous_position[0]:
            reward -= 1  # Bestrafung für Rückschritt

        # Bestrafung für Bewegung nur entlang der Y-Achse
        if kirby_x == self.previous_position[0] and kirby_y != self.previous_position[1]:
            reward -= 2  # Bestrafung für vertikale Bewegung ohne horizontalen Fortschrit

        # Erkennen, ob der Bosskampf begonnen hat
        if boss_health > 0:  # Bosskampf beginnt, wenn der Boss Gesundheit hat
            reward += 500  # Hohe Belohnung für Levelabschluss
            self.kirby.reset_game()
            level_complete = True

        # Bestrafung für Gesundheitsverlust
        if current_health < self.previous_health:
            health_loss = self.previous_health - current_health
            reward -= 10 * health_loss  # Bestrafung für jeden verlorenen Gesundheitsbalken

        # Bestrafung für Verlust eines Lebens
        if current_lives < self.previous_lives:
            reward -= 50  # Starke Bestrafung
            level_complete = True  # Level wird beendet, wenn ein Leben verloren wird

        # Aktualisierung der Zustände
        self.previous_position = (kirby_x, kirby_y)
        self.previous_health = current_health
        self.previous_lives = current_lives
        self.previous_score = current_score

        return reward, level_complete



    def _check_done(self):
        return self.kirby.game_over()
