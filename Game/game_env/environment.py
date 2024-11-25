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
        self.previous_boss_health = None
        self.previous_level_progress = None
        self.previous_game_state = 1
        self.y_axis_steps = 0  # Neu: Anzahl der Schritte nur auf der y-Achse

    def IsBossActive(self):
        #Prüfe ob Boss aktiv ist
        current_boss_health = self.pyboy.memory[0xD093]
        return current_boss_health > 0

    def reset(self):     
        self.kirby.reset_game()
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
        self.previous_boss_health = self.pyboy.memory[0xD093]
        self.previous_level_progress = self._calculate_level_progress()
        self.previous_game_state = self.pyboy.memory[0xD02C] = 1
        self.y_axis_steps = 0  # Zurücksetzen der y-Achsen-Schritte
        
        return np.stack(self.frame_stack, axis=0)

    def step(self, action):
        self._perform_action(action)
        self.pyboy.tick()

        # Update the frame stack
        self.frame_stack.append(self.pyboy.game_area())

        # Erstelle die Beobachtung (4 gestapelte Frames)
        next_state = np.stack(self.frame_stack, axis=0)

        # Belohnung berechnen und überprüfen, ob das Level abgeschlossen ist
        reward, level_complete, life_lost = self._calculate_reward()

        # Behandle Boss-Erreichen und Lebensverlust
        if level_complete:
            return next_state, reward, True, {"level_complete": True}
        
        if life_lost:
            return next_state, reward, True, {"life_lost": True, "level_complete": False}

        # Standard-Fall: Weitertrainieren
        done = self._check_done()  # Kirby verliert alle Leben oder das Spiel ist vorbei
        return next_state, reward, done, {"level_complete": False,  "life_lost": False}
    
    
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
        scx = (screen_x_position * 16 + kirby_x_position)
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

        # 1. Boss besiegt
        if current_boss_health == 0 and self.previous_boss_health > 0:
            print("Boss besiegt")
            #level_complete = True
            reward += 10000

        # 2. Schaden am Boss
        if current_boss_health < self.previous_boss_health:
            print("Boss schaden gemacht")
            reward += 1000

        # 3. Verlust von einem Leben
        if current_lives < self.previous_lives:
            reward -= 3500
            print("Auaa -3500")
            self.kirby.reset_game()
            life_lost = True

        # 4. Verlust von HP 
        if current_health < self.previous_health:
            reward -= 10

        # 5. Bewegung nach links
        if current_level_progress != self.previous_level_progress and kirby_x_position == 68:
            #print("nach links")
            reward -= 5

        # 6. Bewegung nach rechts
        if kirby_x_position == 76:
            #print("nach rechts")
            #print(kirby_x_position)
            reward += 10
        
        # 7. Kiry steht still
        if current_level_progress == self.previous_level_progress:
            reward-= 1

        # 8.Kirby steht still auf der x-Achse
        if current_position[0] == self.previous_position[0]:
            # Prüfen, ob Bewegung nur auf der y-Achse stattfindet
            if current_position[1] != self.previous_position[1]:
                self.y_axis_steps += 1
                
            else:
                self.y_axis_steps = 0  # Zurücksetzen, wenn er sich bewegt

            # Bestrafe, wenn er zu viele Schritte nur auf der y-Achse gemacht hat
            if self.y_axis_steps > 200:
                reward -= 30
                self.y_axis_steps = 0  # Zurücksetzen nach Bestrafung
        else:
            self.y_axis_steps = 0  # Zurücksetzen, wenn er sich horizontal bewegt

        # 9. Punktzahl erhöhen
        if current_score > self.previous_score:
            print("Gegner erledigt")
            reward += 5

        # 10. Warpstar erreicht
        if current_health > 0 and current_game_state == 6 and self.previous_game_state != 6:
            print(current_game_state)
            print("Warpstar erreicht")
            level_complete = True
            reward += 10000
            #self.kirby.reset_game()
            

        # Aktualisierung der Zustände
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
