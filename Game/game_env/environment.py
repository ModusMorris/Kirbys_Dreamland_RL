from pyboy import PyBoy

class KirbyEnvironment:
    def __init__(self, rom_path: str):
        self.pyboy = PyBoy(rom_path, window="SDL2", sound=False)
        self.kirby = self.pyboy.game_wrapper
        self.kirby.start_game()  # Startet das Spiel
        
    def get_state(self):
        """Gibt den aktuellen Zustand von Kirby zurück."""
        current_position = self.get_kirby_position_from_tiles() or (0, 0)
        state = [
            self.kirby.score,
            self.kirby.health,
            self.kirby.lives_left,
            current_position[0],
            current_position[1]
        ]
        return state    

    def get_kirby_position_from_tiles(self):
        """Sucht in den Tiles nach einem charakteristischen Zeichen für Kirby."""
        game_area = self.kirby.game_area()  # Holen des 2D-Arrays mit Tiles
        for y, row in enumerate(game_area):
            for x, tile in enumerate(row):
                if tile == 204:  # Beispiel-ID für Kirby-Tile; dies muss angepasst werden
                    return (x, y)  # Gibt die Position von Kirby zurück
        return None  # Wenn Kirby nicht gefunden wurde

    def calculate_reward(self):
        reward = 0
        current_score = self.kirby.score
        current_health = self.kirby.health
        current_lives = self.kirby.lives_left
        current_position = self.get_kirby_position_from_tiles()

        if current_position and self.previous_position:
            # Belohnung für Fortschritt nach rechts
            if current_position[0] > self.previous_position[0]:
                reward += 5
            elif current_position[0] < self.previous_position[0]:  # Bestrafung für Rückwärtsbewegung
                reward -= 5

        # Restliche Berechnungen unverändert
        if self.kirby.score > self.previous_score:
            reward += (self.kirby.score - self.previous_score) * 10

        if current_health < self.previous_health:
            reward -= (self.previous_health - current_health) * 10

        if current_lives < self.previous_lives:
            reward -= 50

        self.previous_score = current_score
        self.previous_health = current_health
        self.previous_lives = current_lives
        self.previous_position = current_position

        return reward
    def is_near_tile(self, position, tile_position, threshold=5):
        return abs(position[0] - tile_position[0]) <= threshold and abs(position[1] - tile_position[1]) <= threshold

    def is_near_dangerous_tile(self, position):
        dangerous_tiles = [(5, 10), (6, 11)]
        for tile_position in dangerous_tiles:
            if self.is_near_tile(position, tile_position):
                return True
        return False

    def step(self, action):
        self.pyboy.send_input(action)
        self.pyboy.tick()
        reward = self.calculate_reward()
        done = self.kirby.game_over()
        return reward, done

    def reset(self):
        self.kirby.reset_game()
        self.previous_score = self.kirby.score
        self.previous_health = self.kirby.health
        self.previous_lives = self.kirby.lives_left
        self.previous_position = self.get_kirby_position_from_tiles()

    def close(self):
        self.pyboy.stop()
