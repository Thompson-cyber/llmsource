# Adapted from code generated by chatGPT using the prompt "code an example of roulette selection for me". This was generated on February 13th, 2024.
# All code, excluding the adapted portions of the selection code, is my own.

# imports
from math import sin
import random
import numpy as np

# fitness function as defined in the homework
def fitness(x : float) -> float:
    return 4 + (2*x) + (2*sin(20*x)) - 4 * (x**2)

# roulette selection function
def selection(group : list[float]) -> list[float]:
    TF : float = sum(fitness(x) for x in group) # total fitness of the group
    select_prob : list[float] = [fitness(x) / TF for x in group] # proportional probability of each selection for the group
    cum_prob : list[float] = [sum(select_prob[:i+1]) for i in range(len(select_prob))] # cumulative probability of each selection for the group

    new_group : list[float] = [] # new group
    for _ in range(10):
        r : float = random.random() # random value
        for i, prob in enumerate(cum_prob):
            if r <= prob:
                new_group.append(group[i]) # if the random value is less than the cumulative probability, put it in the next group
                break
    
    return new_group

# crossover function
def cross(x,y,a) -> float:
    return x * x + (1 - a) * y

# main search function
def search(crossover : bool) -> None:
    group : list[float] = [0.01*k for k in range(10)] # create the N = 10 individuals

    # run for 100 generations
    for _ in range(100):
        new_group : list[float] = [] # create the list representing the next gen's group

        # iterate over each individual in the group
        for i in range(10):
            selected : list[float] = selection(group) # run selection on the current group

            # if specified, use crossover on the given value.
            if not crossover:
                offspring : float = selected[i] # simply set the "offspring" to the selcted value if NOT doing crossover
            else:
                a : float = random.uniform(0,1)
                x : float = random.choice(selected)
                y : float = random.choice(selected)
                offspring : float = cross(x,y,a) # set the selected value to the result of a crossover method

            r : float = random.random() # get a random probability
            
            if r <= 0.3: # x - epsilon with probability of 0.3
                new_group.append(np.clip(offspring - 0.01, 0, 1))
            elif r <= 0.6: # x + epsilon with probability of 0.3
                new_group.append(np.clip(offspring + 0.01, 0, 1))
            else: # copy with probability of 0.4
                new_group.append(offspring)
            
        group = new_group # replace the old group with the new group
    
    # print the results.
    bestN = max(group, key=fitness)
    print(f"Best N: {bestN}.")
    print(f"Best Fitness: {fitness(bestN)}")
if __name__ == "__main__":
    print("========WITHOUT CROSSOVER========")
    search(False) # part 1
    print("========WITH CROSSOVER========")
    search(True) # part 2