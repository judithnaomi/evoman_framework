# Assignment 1 Evolutionary Computing
# Freek Stroes, Thijs Roukens, Sebastian Smit, Judith Schermer
# 27 September 2019

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

experiment_name = 'assignment_specialist1'      # Assignment task 1
enemy = 2                                       # Set correct enemy to run in this solution
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
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

################################### OWN PART ###########################################

population = {} # population is a dictionary with keys individual IDs
                #and values are tuples where the first element is the fitness value and 
                #the second element is the list neutral network weights between -1 and 1.
                #e.g. population['1'] = (0.89, [-0.433434,-0.4324234,0.58532, ..., ], mutation_sigma) here, individual
                # 1 has fitness 0.89 and [-0.4333434,...] are the neural network weights.

id_individual = 0 # the id of the newest individual    


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
        for i in range(0, n_vars):
            individual.append(rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1

        initial_population.append(individual)

    return np.array(initial_population)

def fitness(individual):
    return simulation(env, individual)

def ordered_population(population):
    # returns the population as a list of tuples i.e. 
    # [(ID,[fitness,list_of_values]), (ID,[fitness,list_of_values]),....]
    # note: fitness in decreasing order, so starting with fittest individual
    return sorted(population.items(), key = lambda kv:kv[1][0], reverse=True) 
    # e.g. kv[1]=(fitness,individual). kv[1][0] is the fitness and fitness is used to sort the list. 


def mate():
    newborns = []
    global population
    pop = ordered_population(population)
    random_numbers = list(range(0, len(population))) #generates a list counting from 0 up to and including n_pop -1
    rand.shuffle(random_numbers) #shuffle the numbers

    for i in range(0, len(pop), 5): #in steps of 5
        five_random = [random_numbers[i], random_numbers[i + 1], random_numbers[i + 2], random_numbers[i + 3], random_numbers[i + 4]] #makes a list with 5 random numbers
        five_random.sort() # sorts these number from low to high

        #the first mating pair; chosing the lowest random number automatically gives the one with the highest fitness becaus pop is ordered on fitness
        key_dad1, value_dad1 = pop[five_random[0]]
        key_mom1, value_mom1 = pop[five_random[1]]

        # the second mating pair
        key_dad2, value_dad2 = pop[five_random[2]]
        key_mom2, value_mom2 = pop[five_random[3]]

        #creating the children
        genotype_child1 = (value_mom1[1] + value_dad1[1]) / 2
        genotype_child2 = (value_mom2[1] + value_dad2[1]) / 2

        gene_mutations1 = [(rand.random() - .5) * .1 for x in
                          range(n_vars)]  # creates some random numbers between -.1 and .1
        gene_mutations2 = [(rand.random() - .5) * .1 for x in
                           range(n_vars)]

        genotype_child1 = keep_within_boundaries(np.add(gene_mutations1, genotype_child1), -1, 1)
        genotype_child2 = keep_within_boundaries(np.add(gene_mutations2, genotype_child2), -1, 1)

        #append the children to the newborns
        newborns.append(genotype_child1)
        newborns.append(genotype_child2)# offspring is added to the population (born).
    return newborns


def select_survivors():  # this method kills a specified number of the least fit individual
    global population
    n_deaths = int(len(population)/5)*2 # number of individuals that dies each generation
    deaths = ordered_population(population)[::-1][:n_deaths] # we take the first n_deaths elements
    #of the inverted population list: the weakest individuals
   
    for key,value in deaths:
        del population[key]
        print("strategy " + str(key) + " died")

def add_individuals_to_population(new_individuals):
    for individual in new_individuals:
        global id_individual, population
        population[id_individual] = [fitness(individual), individual]
        id_individual = id_individual + 1
        print("individual " + str(id_individual) +" added")

def print_ordered_population_nicely():
    for key,value in ordered_population(population):
        print(key,value[0]), #print only the individual ID and the fitness

def difference(generation, previous_generation): #computes difference in fitnesses between 
# two generations using sum of differences between ranked individuals
        differences = []
        
        ordered_1 = ordered_population(generation)
        ordered_2 = ordered_population(previous_generation)

        for i in range(len(generation)):

            key1, value1 = ordered_1[i]
            key2, value2 = ordered_2[i]
            difference_in_fitness = abs(value1[0]-value2[0])
            differences.append(difference_in_fitness)

        return sum(differences)


def get_max_fitness(population1):
    max_value = -1000
    population = ordered_population(population1)
    for i in range(len(population)):
        key1, value1 = population[i]
        if value1[0] > max_value:
            max_value = value1[0]
        else:
            max_value = max_value
    return max_value

def write_results(run, generations):
    if run == 1:
        file_results = open(experiment_name+'/results.txt', 'w')
        file_results.write('Tested Enemy: ' + str(enemy) + '\n')
    else:
        file_results = open(experiment_name+'/results.txt', 'a')

    file_results.write('RUN ' + str(run) + '\n\n')
    file_results.write('# of generations: ' + str(len(generations)) + '\n')

    file_results.write('Best fitness: ' + str(ordered_population(generations[-1])[0][1][0]) + '\n\n')

    average_fitness = []
    std_fitness = []

    for generation in generations:
        fitness_array_of_generation = []
        # we sum the fitnesses of all individuals in this generation
        ordered_pop = ordered_population(generation)
        n_pop = len(generations[0])
        for i in range(n_pop): 
            individual = ordered_pop[i]
            ind_fitness, weights = individual[1]
            fitness_array_of_generation.append(ind_fitness)
        # we append the average fitness for this generation 
        average_fitness.append(np.mean(fitness_array_of_generation))
        # we append the standard deviation in the fitness for this generation 
        std_fitness.append(np.std(fitness_array_of_generation))

    # we compute the average fitness of all generations, by averaging the averages... same for standard deviation...
    file_results.write('Average of the fitness over the generations of this run: ' + str(np.mean(average_fitness)) + '\n')
    file_results.write('Average standard deviation of the fitness over the generations of this run: ' + str(np.mean(std_fitness)) + '\n\n')
    file_results.close()

def perform_run(n_pop, difference_threshold):

    print('Generation 1')

    pioneers = spawn(n_pop)
    add_individuals_to_population(pioneers)
    generations = []
    generations.append(population.copy())
    no_improvement = 0 # amount of generations that have not been improving

    old_best_fitness = get_max_fitness(population)


    # if there is no significant improvement after .. generations, then terminate 
    while no_improvement < difference_threshold: 

        newborns = mate()        
        add_individuals_to_population(newborns)
        select_survivors()
        generations.append(population.copy())

        print("Generation" + str(len(generations)))
        print("Difference:")
        print(difference(generations[-1], generations[-2])) # compare two last generations
        print_ordered_population_nicely()

        best_fitness = get_max_fitness(population)
        if best_fitness > old_best_fitness:
            old_best_fitness = best_fitness
            no_improvement = 0
        else:
            no_improvement += 1
        print('\nNo improvement: '+str(no_improvement))

    return generations

def main(n_pop = 100, difference_threshold = 3, n_runs = 10):
      
    for run in range(n_runs):
        
        print('RUN: ' + str(run))

        generations = perform_run(n_pop, difference_threshold)

        write_results(run, generations)

        global population
        population = {}

    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

main(15,3,1)