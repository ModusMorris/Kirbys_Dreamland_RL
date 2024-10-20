import random

class RandomAgent:
    def __init__(self, button_list):

        self.button_list = button_list

    def select_action(self):

        return (random.choice(self.button_list))
    
    #random.choice(self.button_list)