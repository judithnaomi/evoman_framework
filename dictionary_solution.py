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
npop = 4
gens = 30
mutation = 0.2
last_best = 0

################################### OWN PART ###########################################

n_generations = 6
n_deaths = 2  # number of individuals that dies each generation
id_individual = 0 # the ids of the new individuals only increase

population = {}  # population is a dictionary with keys individual IDs
#and values are tuples where the first element is the fitness value and 
#the second element is the list of nvars real numbers
#between -1 and 1.
#e.g. population['1'] = (0.89, [-0.433434,-0.4324234,0.58532, ..., ]) here, individual
# 1 has fitness 0.89 and [-0.4333434,...] etc is its representation.

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def spawn(size):  # a prespecified number (''size'') of individuals is spawned
    initial_population = []
    for times in range(0, size):  # these individuals are characterized by their genotype
        individual = []
        for i in range(0, n_vars):
            individual.append(rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1

        initial_population.append(individual)

    return np.array(initial_population)

def fitness(individual):
    return simulation(env, individual)

# returns the population as a list of triples, i.e. 
# [(ID,fitness,list_of_values), (ID,fitness,list_of_values),....]
def ordered_population():
    return sorted(population.items(), key = lambda kv:kv[1][0], reverse=True) #kv[1] is the value of the keyvalue pair
#kv, i.e. kv[1]=(fitness,individual). kv[1][0] is the fitness.


def mate():

    newborns = []
    pop = ordered_population()

    for i in range(0,len(pop),2): #every iteration, i increases with 2
        key_dad, value_dad = pop[i]
        key_mom, value_mom = pop[i+1]
        # we want them to be monogamous and to mate in pairs
        
        genotype_child = (value_mom[1] + value_dad[1]) / 2
     
       # the numbers of the  fittest with their subsequent individual are averaged to create a baby always skipping one (monogamous).
        gene_mutations = [(rand.random() - .5) * .1 for x in range(n_vars)]  # creates some random numbers between -.1 and .1
       
        genotype_child = np.add(gene_mutations, genotype_child)
      
        # the random numbers are added to the offsprings list to create mutations in its genome.

        newborns.append(genotype_child)  # offspring is added to the population (born).
    return newborns

def selection(n_deaths):  # this method kills a specified number of the least fit individuals
    deaths = ordered_population()[::-1][:n_deaths] #we take the first n_deaths elements
    #of the inverted population list, the weakest individuals
   
    for key,value in deaths:
        del population[key]
        print("strategy " + str(key) + " died")

def add_individuals_to_population(new_individuals):

    for individual in new_individuals:
        global id_individual 
        population[id_individual] = [fitness(individual), individual]
        id_individual = id_individual + 1
        print("individual " + str(id_individual) +" added")

def print_ordered_population_nicely():
    for key,value in ordered_population():
        print(key,value[0]), #print only the individual ID and the fitness

############ INITIALIZE POPULATION ###################

pioneers = spawn(npop)

add_individuals_to_population(pioneers)

############ MAIN PROGRAM ########################

for i in range(n_generations):
    print("Generation" + str(i))

    newborns = mate()
    add_individuals_to_population(newborns)
    
    selection(n_deaths)
    print_ordered_population_nicely()

 



