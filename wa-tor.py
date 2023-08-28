# Erik Cabrera
# Wa-Tor
# 24.08.2023

# !pip3 install mesa matplotlib IPython numpy --quiet

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from matplotlib import animation
from enum import Enum
from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt

class CellType(Enum):
    EMPTY = 0
    FISH = 1
    SHARK = 2

# Fish Agent
class Fish(Agent):
    def __init__(self, unique_id, model, energy):
        super().__init__(unique_id, model)
        self.energy = energy
        self.fertility_counter = 0

    def step(self):
        self.move()
        self.energy -= 1
        self.reproduce()
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)

    def reproduce(self):
        self.fertility_counter += 1
        if self.fertility_counter >= self.model.fish_fertility_threshold:
            offspring = Fish(self.model.next_id(), self.model, self.model.fish_initial_energy)
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.add(offspring)
            self.fertility_counter = 0

# Shark Agent
class Shark(Agent):
    def __init__(self, unique_id, model, energy):
        super().__init__(unique_id, model)
        self.energy = energy
        self.fertility_counter = 0

    def step(self):
        self.move_and_eat()
        self.reproduce()
        self.energy -= 1
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move_and_eat(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False) 
        fish_positions = [step for step in possible_steps if
                          self.model.grid.get_cell_list_contents([step]) and
                          isinstance(self.model.grid.get_cell_list_contents([step])[0], Fish)]
        if fish_positions:
            new_position = self.random.choice(fish_positions)
            fish = self.model.grid.get_cell_list_contents([new_position])[0]
            self.model.grid.remove_agent(fish)
            self.model.schedule.remove(fish)
            self.model.grid.move_agent(self, new_position)
            self.energy += self.model.shark_energy_from_fish
        else:
            empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
            if empty_steps:
                new_position = self.random.choice(empty_steps)
                self.model.grid.move_agent(self, new_position)


    def reproduce(self):
        self.fertility_counter += 1
        if self.fertility_counter >= self.model.shark_fertility_threshold:
            offspring = Shark(self.model.next_id(), self.model, self.model.shark_initial_energy)
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.add(offspring)
            self.fertility_counter = 0

def compute_grid(model):
    return model.get_grid()

# War-Tor Model
class WaTorModel(Model):
    def find_empty_cell(self):
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        while not self.grid.is_cell_empty((x, y)):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
        return x, y
    
    def __init__(self, width, height, N_fish, N_sharks, fish_initial_energy, shark_initial_energy,
                 fish_fertility_threshold, shark_fertility_threshold, shark_energy_from_fish):
        self.num_agents = N_fish + N_sharks
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Grid": compute_grid})
        self.fish_initial_energy = fish_initial_energy
        self.shark_initial_energy = shark_initial_energy
        self.fish_fertility_threshold = fish_fertility_threshold
        self.shark_fertility_threshold = shark_fertility_threshold
        self.shark_energy_from_fish = shark_energy_from_fish
        self.current_id = 0

        for _ in range(N_fish):
            x, y = self.find_empty_cell()
            fish = Fish(self.next_id(), self, fish_initial_energy)
            self.grid.place_agent(fish, (x, y))
            self.schedule.add(fish)

        for _ in range(N_sharks):
            x, y = self.find_empty_cell()
            shark = Shark(self.next_id(), self, shark_initial_energy)
            self.grid.place_agent(shark, (x, y))
            self.schedule.add(shark)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def get_grid(self):
        grid = np.zeros((self.grid.width, self.grid.height))
        for cell in self.grid.coord_iter():
            content, (x, y) = cell
            if content:
                if isinstance(content[0], Shark):
                    grid[x][y] = CellType.SHARK.value
                elif isinstance(content[0], Fish):
                    grid[x][y] = CellType.FISH.value
        return grid

# Initial parameters
width = 60  
height = 85
N_fish = 140
N_sharks = 45
fish_initial_energy = 20
shark_initial_energy = 4
fish_fertility_threshold = 6
shark_fertility_threshold = 10
shark_energy_from_fish = 8

MAX_ITERATIONS = 100

model = WaTorModel(width, height, N_fish, N_sharks, fish_initial_energy, shark_initial_energy,
                   fish_fertility_threshold, shark_fertility_threshold, shark_energy_from_fish)

for i in range(MAX_ITERATIONS):
    model.step()

all_grid = model.datacollector.get_model_vars_dataframe()


fig, axis = plt.subplots(figsize=(10,6))
axis.set_xticks([])
axis.set_yticks([])
patch = plt.imshow(all_grid.iloc[0]["Grid"], cmap='cividis')


def animate(i):
    patch.set_data(all_grid.iloc[i]["Grid"])

anim_html = animation.FuncAnimation(fig, animate, frames=len(all_grid)).to_jshtml()
HTML(anim_html)

#__END__ 
