# Assignment 1 Evolutionary Computing
# Freek Stroes, Thijs Roukens, Sebastian Smit, Judith Schermer
# 18 September 2019


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
npop = 100
gens = 30
mutation = 0.2
last_best = 0

################################### OWN PART ###########################################

#n_generations = 6
difference_threshold = 0 

n_deaths = int(npop/5)*2 # number of individuals that dies each generation

id_individual = 0 # the ids of the new individuals only increase

population = {}  # population is a dictionary with keys individual IDs
#and values are tuples where the first element is the fitness value and 
#the second element is the list neutral network weights between -1 and 1.
#e.g. population['1'] = (0.89, [-0.433434,-0.4324234,0.58532, ..., ], mutation_sigma) here, individual
# 1 has fitness 0.89 and [-0.4333434,...] are the neural network weights.

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def keep_within_boundaries(ls, lower, upper):
    # if any number in the list is higher than the highest allowed value, change it to highest value.
    # if any number in the list is lower than the lowest allowed value, change it to the lowest value. 
    for i in range(len(ls)):
        if ls[i] < lower:
            ls[i] = lower
        elif ls[i] > upper:
            ls[i] = upper
    return ls   

def spawn(size):  # a prespecified number (''size'') of individuals is spawned
    initial_population = []


    for times in range(0, size):  # these individuals are characterized by their genotype
        individual = []
        mutation_individual = rand.uniform(-0.5, 0.5)

        for i in range(0, n_vars):
            individual.append(rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1

        initial_population.append((individual, mutation_individual))


    return initial_population

def fitness(individual):
    return simulation(env, individual)

# returns the population as a list of triples, i.e. 
# [(ID,fitness,list_of_values), (ID,fitness,list_of_values),....]
# note: fitness in decreasing order, so starting with fittest individual
def ordered_population(population):
    return sorted(population.items(), key = lambda kv:kv[1][0], reverse=True) 
    # e.g. kv[1]=(fitness,individual,mutation_sigma). kv[1][0] is the fitness and fitness is used to sort the list. 


def mate():
    newborns = []
    pop = ordered_population(population)
    random_numbers = list(range(0, npop )) #generates a list counting from 0 up to and including npop -1
    rand.shuffle(random_numbers) #shuffle the numbers

    for i in range(0, len(pop), 5):
        five_random = [random_numbers[i], random_numbers[i + 1], random_numbers[i + 2], random_numbers[i + 3], random_numbers[i + 4]] #makes a list with 5 random numbers
        five_random.sort() # sorts these number from low to high

        #the first mating pair; chosing the lowest random number automatically gives the one with the highest fitnes becaus pop is ordered on fitness
        key_dad1, value_dad1 = pop[five_random[0]]
        key_mom1, value_mom1 = pop[five_random[1]]

        # the second mating pair
        key_dad2, value_dad2 = pop[five_random[2]]
        key_mom2, value_mom2 = pop[five_random[3]]

        #creating the children
        genotype_child1 = (np.array(value_mom1[1])  + np.array(value_dad1[1])) / 2
        genotype_child2 = (np.array(value_mom2[1]) + np.array(value_dad2[1])) / 2

        mutation_child1 = rand.uniform(-0.05, 0.05)
        mutation_child2 = rand.uniform(-0.05, 0.05)

        mutation_mom1 = value_mom1[2]
        mutation_dad1 = value_dad1[2]
        mutation_child1 = (mutation_mom1 + mutation_dad1)/2 + mutation_child1

        mutation_mom2 = value_mom2[2]
        mutation_dad2 = value_dad2[2]
        mutation_child2 = (mutation_mom2 + mutation_dad2)/2 + mutation_child2


        gene_mutations1 = [mutation_child1 for x in  range(n_vars)]  # creates some random numbers between -.1 and .1
        gene_mutations2 = [mutation_child2  for x in range(n_vars)]
   

        genotype_child1 = keep_within_boundaries(np.add(gene_mutations1, genotype_child1), -1, 1)
        genotype_child2 = keep_within_boundaries(np.add(gene_mutations2, genotype_child2), -1, 1)

        #append the children to the newborns
        newborns.append((genotype_child1,mutation_child1))
        newborns.append((genotype_child2,mutation_child2))# offspring is added to the population (born).
    return newborns




def perform_selection(n_deaths):  # this method kills a specified number of the least fit individuals
    deaths = ordered_population(population)[::-1][:n_deaths] #we take the first n_deaths elements
    #of the inverted population list, the weakest individuals
   
    for key,value in deaths:
        del population[key]
        print("strategy " + str(key) + " died")

def add_individuals_to_population(new_individuals):

    for individual, mutation_sigma in new_individuals:

  #  for individual in new_individuals:
        global id_individual 
        population[id_individual] = [fitness(np.array(individual)), individual, mutation_sigma]
        id_individual = id_individual + 1
        print("individual " + str(id_individual) +" added")

def print_ordered_population_nicely():
    for key,value in ordered_population(population):
        print(key,value[0]), #print only the individual ID and the fitness

def difference(generation, previous_generation):
        differences = []
        
        ordered_1 = ordered_population(generation)
        ordered_2 = ordered_population(previous_generation)

        for i in range(len(generation)):

            key1, value1 = ordered_1[i]
            key2, value2 = ordered_2[i]
            difference_in_fitness = abs(value1[0]-value2[0])
            differences.append(difference_in_fitness)

        
        #return np.exp((sum([x**2 for x in differences]))) 
        #difference measured by the sum of squares
        return sum(differences)
      

############ INITIALIZE POPULATION ###################

pioneers = spawn(npop)
#voeg een tweede getal toe aan pioneers. 
#dan is new_individuals dus een lijst van paren 

add_individuals_to_population(pioneers)

############ MAIN PROGRAM ########################

generations = []
generations.append(population.copy())

newborns = mate()
add_individuals_to_population(newborns)
perform_selection(n_deaths)
generations.append(population.copy())

#for i in range(n_generations):

i=1

while difference(generations[i],generations[i-1])>difference_threshold:
    i = i + 1
    print("Generation" + str(i))

    newborns = mate()
    add_individuals_to_population(newborns)
    
    perform_selection(n_deaths)

    generations.append(population.copy())
    print("Difference:")
    print(difference(generations[i], generations[i-1]))
    print_ordered_population_nicely()



fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
