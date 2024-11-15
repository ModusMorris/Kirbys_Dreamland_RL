import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from pyboy.utils import WindowEvent
import pyboy.plugins.game_wrapper_kirby_dream_land as gw
from PIL import Image

class KirbyEnvironment(gym.Env):
    def __init__(self, rom_path="Kirbys_Dream_Land.gb"):
        super(KirbyEnvironment, self).__init__()
        self.pyboy = PyBoy(rom_path, window="SDL2")  # Verwende SDL2 für die GUI
        self.kirby = self.pyboy.game_wrapper  # Zugriff auf den Game Wrapper
        self.kirby.start_game()  # Starte das Spiel, um Kirby zu initialisieren

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 20, 16), dtype=np.float32)
        
        # Emulationsgeschwindigkeit auf unendlich setzen, um schnelleres Training zu ermöglichen
        self.pyboy.set_emulation_speed(0)  # 0 bedeutet maximale Geschwindigkeit

        # Frame stack für die letzten 4 Frames
        self.frame_stack = deque(maxlen=4)
        self.previous_position = (-1, -1)  # Initialisiere die vorherige Position
        self.previous_health = self.kirby.health  # Gesundheitsstatus von Kirby
        self.previous_lives = self.kirby.lives_left  # Anzahl der verbleibenden Leben
        self.reset()

    def reset(self):
        # Start und Reset des Spiels
        self.kirby.reset_game()
        self.frame_stack.clear()  # Leere den Frame Stack
        initial_frame = self._get_screen_image()
        for _ in range(4):
            self.frame_stack.append(initial_frame)
        self.previous_position = (-1, -1)  # Zurücksetzen der Position beim Reset
        self.previous_health = self.kirby.health  # Gesundheitsstatus zurücksetzen
        self.previous_lives = self.kirby.lives_left  # Anzahl der Leben zurücksetzen
        return self._get_observation()

    def step(self, action, skip_frames=4):
        for _ in range (skip_frames):
            self._perform_action(action)
            self.pyboy.tick()  # Ein Frame weiter
        reward = self._calculate_reward()
        done = self._check_done()

        # Fange das neue Frame ab und füge es zum Stack hinzu
        new_frame = self._get_screen_image()
        self.frame_stack.append(new_frame)

        # Prüfen, ob Kirby Schaden genommen hat oder ein Leben verloren hat
        current_health = self.kirby.health
        current_lives = self.kirby.lives_left

        if current_health < self.previous_health:
            # Bestrafung für das Verlieren von Gesundheit
            reward -= 10.0
            done = True  # Episode endet, wenn Kirby Schaden erleidet

        if current_lives < self.previous_lives:
            # Bestrafung für das Verlieren eines Lebens
            reward -= 50.0
            done = True  # Episode endet, wenn Kirby ein Leben verliert

        # Aktualisiere den Gesundheitsstatus und die Anzahl der Leben
        self.previous_health = current_health
        self.previous_lives = current_lives

        # Rückgabe von vier Werten: Beobachtung, Belohnung, ob Spiel vorbei ist und Info
        observation = self._get_observation()
        return observation, reward, done, {}

    def _get_screen_image(self):
        # Verwende pyboy.screen.image, um den Bildschirm abzurufen und konvertiere in Graustufen
        screen_image = self.pyboy.screen.image.convert("L")
        resized_image = screen_image.resize((16, 20), Image.LANCZOS)
        screen_array = np.array(resized_image, dtype=np.float32) / 255.0
        return screen_array

    def _get_observation(self):
        # Beobachtungen umfassen nur die letzten 4 Frames als NumPy Array
        return np.stack(self.frame_stack, axis=0)

    def _perform_action(self, action):
        # Definieren von Aktionen basierend auf PyBoy's Eingabesystem
        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
        elif action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
        elif action == 2:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        elif action == 3:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
        elif action == 4:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
        elif action == 5:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
        self.pyboy.tick()
        
    def _find_warpstar_position(self):
        # Beispielhafte Implementierung: Erkenne den Warpstar anhand bestimmter Kriterien im Frame
        frame = self.frame_stack[-1]  # Verwende das neueste Frame zur Analyse
        warpstar_threshold = 0.9  # Setze einen Threshold für die Erkennung des Warpstars (helles Pixelmuster o.ä.)
        warpstar_positions = np.argwhere(frame > warpstar_threshold)  # Suche nach auffällig hellen Pixeln
        
        if warpstar_positions.size > 0:
            # Geben Sie die durchschnittliche Position des Warpstars zurück
            avg_position = warpstar_positions.mean(axis=0)
            return tuple(avg_position)  # Konvertiere zu Tupel (row, col)
        
        return (-1, -1)  # Rückgabe ungültiger Position, wenn der Warpstar nicht erkannt wird
            
    def _calculate_reward(self):
        current_position = self._find_kirby_position()
        warpstar_position = self._find_warpstar_position()
        reward = 0.0

        # Fortschrittsbelohnung: Bewegung nach rechts
        if current_position[1] > self.previous_position[1]:
            reward += 10.0  # Stärkere Belohnung für Bewegung nach rechts

        # Bestrafung für Stillstand (horizontal)
        if current_position == self.previous_position:
            reward -= 5.0  # Bestrafung für Stillstand

        # Bestrafung für Bewegung nach oben oder Fliegen an eine Wand
        if current_position[0] == 0:
            reward -= 10.0  # Bestrafung für Fliegen an die Wand

        # Zusätzliche Belohnung, wenn Kirby den Warpstar erreicht
        if warpstar_position != (-1, -1) and current_position == warpstar_position:
            reward += 1000.0  # Sehr hohe Belohnung für das Erreichen des Warpstars
            done = True  # Episode beenden

        # Aktualisiere die Zustände für die nächste Runde
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_position = current_position

        return reward


        return reward
    def _find_kirby_position(self):
        # Benutzermethode, um die Position von Kirby zu schätzen
        frame = self.frame_stack[-1]  # Verwende das neueste Frame zur Analyse
        threshold = 0.8  # Passen Sie dies an, um Kirby anhand der Pixelintensität zu erkennen
        possible_positions = np.argwhere(frame > threshold)
        if possible_positions.size > 0:
            # Geben Sie die durchschnittliche Position zurück, wenn mögliche Kirby-Pixel erkannt werden
            avg_position = possible_positions.mean(axis=0)
            return tuple(avg_position)  # Konvertiere zu Tupel (row, col)
        return (-1, -1)  # Rückgabe ungültiger Position, wenn Kirby nicht erkannt wird

    def _check_done(self):
        # Prüfen, ob das Spiel vorbei ist
        return self.kirby.game_over()
