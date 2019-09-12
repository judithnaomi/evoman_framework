# Assignment 1 Evolutionary Computing
# Freek Stroes, Thijs Roukens, Sebastian Smit, Judith Schermer
# 12 September 2019


# Code until ##### OWN PART ##### is taken from EvoMan FrameWork - V1.0 2016 by Karine Miras

# imports framework
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import random as rand
import numpy as np

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os

experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

n_hidden = 10
n_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 20
gens = 30
mutation = 0.2
last_best = 0

################################### OWN PART ###########################################

n_generations = 3
n_deaths = 10  # number of individuals that dies each generation


# simulation(env,x) is taken from EvoMan FrameWork - V1.0 2016 by Karine Miras
# simulates a game with game strategy x, outputs fitness according to
# default fitness function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def spawn(size):  # a prespecified number (''size'') of individuals is spawned
    population = []
    for times in range(0, size):  # these individuals are characterized by their genotype
        individual = []
        for i in range(0, n_vars):
            individual.append(
                rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1

        population += [individual]
    population = np.array(population)
    return population

def fitness(individual):
    return [simulation(env, individual)]

def compile(fitness, npop):
    population_fitness = zip(fitness,npop.tolist())
    return population_fitness


def order_population_fitness (unorderd_pop):
    population_fitness_tuple = sorted(unorderd_pop)# individuals are sorted based on their respective fitness scores from low to high, hence the reversed iteration
    fitness_scores, population = zip(*population_fitness_tuple)
    population = np.array(list(population[::-1]))
    fitness_scores = np.array(list(fitness_scores[::-1]))
    ordered_population = compile(fitness_scores,population)
    return ordered_population

def mating_population(npop) :
    fitness_scores, population_sorted = zip(*npop)
    return population_sorted

def get_fitness_scores(npop):
    fitness_scores, population_sorted = zip(*npop)
    return fitness_scores


def mate(pop):

    newborns = []
    for i, individual in enumerate(pop):
        if i % 2 == 0 and i < (len(pop) - 1):  # we want them to be monogamous and we do not want an error for the last one.

            offspring = [(g + h) / 2 for g, h in zip(pop[i], pop[i + 1])]  # the numbers of the  fittest with their subsequent individual are averaged to create a baby always skipping one (monogamous).
            gene_mutations = [(rand.random() - .5) * .1 for x in range(n_vars)]  # creates some random numbers between -.1 and .1

            offspring = [sum(x) for x in zip(gene_mutations, offspring)]  # the random numbers are added to the offsprings list to create mutations in its genome.

            newborns += [offspring]  # offspring is added to the population (born).
    return newborns


def selection(pop, n_deaths):  # this method kills a specified number of the least fit individuals

    population = np.array(pop[:-n_deaths])  # the 'n_deaths'- last individuals are removed from the gene pool
    return population

pop = spawn(npop)
fitnes_scores = []
for indivudal in pop:
    fitnes_scores += fitness(indivudal)
compiled = compile(fitnes_scores, pop)
ranked_population = order_population_fitness(compiled)
mating = mating_population(ranked_population)
newborns = np.array(mate(mating))

fitnes_scores_newborns = []
for newborn in newborns:
    fitnes_scores_newborns += fitness(newborn)
compile_newborns = compile(fitnes_scores_newborns,newborns)
total_population = ranked_population + compile_newborns
rank_total_population = order_population_fitness(total_population)
total_population_nof = mating_population(rank_total_population)
new_pop = selection(total_population_nof,n_deaths)
fitness_scores_new_pop = selection(get_fitness_scores(rank_total_population),n_deaths)
compiled_new_pop = compile(fitness_scores_new_pop,new_pop)

for i in range(n_generations):
    new_borns = np.array(mate(new_pop))
    fitnes_scores_newborns = []
    for newborn in newborns:
        fitnes_scores_newborns += fitness(newborn)
    compile_newborns = compile(fitnes_scores_newborns, newborns)
    compiled_new_pop = compiled_new_pop+ compile_newborns
    rank_total_population = order_population_fitness(compiled_new_pop)
    total_population_nof = mating_population(rank_total_population)
    new_pop = selection(total_population_nof, n_deaths)
    fitness_scores_new_pop = selection(get_fitness_scores(rank_total_population), n_deaths)
    compiled_new_pop = compile(fitness_scores_new_pop, new_pop)

print(fitness_scores_new_pop)








