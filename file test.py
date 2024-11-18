import gym
from pyboy import PyBoy
import numpy as np
from collections import deque
from PIL import Image
import os
from pyboy.plugins.game_wrapper_kirby_dream_land import GameWrapperKirbyDreamLand as GW



# Pfad zur ROM-Datei
rom_path = "Game/Kirby.gb"  # Gib hier den Namen oder den Pfad zu deiner ROM an

# PyBoy initialisieren
pyboy = PyBoy(rom_path)

# Emulator starten
print("Emulator gestartet! Schließe das Fenster, um zu beenden.")
while pyboy.tick():
    x= pyboy.memory[0xD053]
    kirbyx = pyboy.memory[0xD05C]
    kirbyy = pyboy.memory[0xD05D]
    lives = pyboy.game_wrapper.lives_left
    game_wrapper = pyboy.game_wrapper.game_area #damit bekomme ich das array worin sich die daten befidnen kirby ist 2 und 18 anfangs auf achse 10
    sprite = pyboy.game_wrapper._sprites_on_screen()
    #print(sprite[2:])
    #game = pyboy.game_wrapper.
    print(game_wrapper)
    pass

# Emulator beenden, nachdem das Fenster geschlossen wurde
pyboy.stop()
