# Assignment 1 Evolutionary Computing
# Freek Stroes, Thijs Roukens, Sebastian Smit, Judith Schermer
# 10 October 2019

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
import csv

enemy = 'undefined'
experiment_name = 'test'  # Assignment task 2
enemies = [4,6,7] # Set correct enemy to run in this solution
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode="yes",
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
n_runs = 10
n_hidden = 10
n_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 10 hidden neurons
npop = 100
id_individual = 1
difference_treshold = 5
sigma = 0.1
mu = 0

# mating parameters
n_downwards = 0 # every generation, number of fittest individuals selected is decreased with n_downwards
n_randparents = 10 # every generation, select n_randparents weaker individuals
n_best_parents = 30 # every generation, select n_best_parents fittest individuals
parents_treshold = 10 # stop with decreasing number of fittest individuals when n_randparents == parents_threshold


# we will use a list containing of lists for our population, each lists represents one individual
# the individual looks like [265 weights, id, fitness , playerlife]

def simulation(x):
    z = np.array(x)
    f, p, e, t = env.play(pcont=z)
    return f, p


def keep_within_boundaries(ls, lower, upper):
    # if any number in the list is higher than the highest allowed value, change it to highest value.
    # if any number in the list is lower than the lowest allowed value, change it to the lowest value.
    for i in range(len(ls)):
        if ls[i] < lower:
            ls[i] = lower
        elif ls[i] > upper:
            ls[i] = upper
    return ls

def spawn():  # a prespecified number (''size'') of individuals is spawned
    initial_population = []
    for times in range(0, npop):  # these individuals are characterized by their genotype
        individual = []
        global id_individual
        for i in range(0, n_vars):
            individual.append(rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1
        fitness, player_life = simulation(individual[0:n_vars])
        individual.append(id_individual)
        individual.append(fitness)
        individual.append(player_life)
        print("individual " + str(individual[n_vars]) + " added")
        id_individual = id_individual + 1
        initial_population.append(individual)
    return initial_population


def add_individuals_to_population(population, new_individuals):
    for individual in new_individuals:
        global id_individual
        fitness, player_life = simulation(individual)
        individual.append(id_individual)
        individual.append(fitness)
        individual.append(player_life)
        id_individual = id_individual +1
        print("individual " + str(individual[n_vars]) + " added")
        population.append(individual)
    return population

def ordered_population(population):
    # note: fitness in decreasing order, so starting with fittest individual
    return sorted(population, key=lambda x:x[n_vars+1], reverse=True)
    # x[n_vars+1] is the fitness and fitness is used to sort the list.


def mate(n_random_parents, population):
    newborns = []
    
    #PARENT SELECTION
    
    pop = ordered_population(population)
    parents = pop[0:n_best_parents] #select best n parents
    left_pop = pop[n_best_parents:]
    random_numbers = list(range(0, len(left_pop)))
    rand.shuffle(random_numbers)

    for i in range(n_random_parents): # random select n_random_parents from leftover population
        random_parent = left_pop[random_numbers[i]]
        parents.append(random_parent)

    ordered_parents = ordered_population(parents)
    for i in range(len(ordered_parents)):
        coin = rand.uniform(0, 1) # flip a coin to see if parent i swap index with random other parent
        if coin <= 0.1:
            number = rand.randint(i, len(ordered_parents)-1)
            ordered_parents[number] = ordered_parents[i]
            ordered_parents[i] = ordered_parents[number]
            
    #CROSS OVER

    for i in range(0,len(ordered_parents),2):
        dad = ordered_parents[i]
        mom = ordered_parents[i+1]

        child1 = (np.array(mom[0:n_vars]) + np.array(dad[0:n_vars]))/2 # taking averages
        child2 = flip_coin_crossover(np.array(mom[0:n_vars]),np.array(dad[0:n_vars]))
        
    #MUTATION

        gene_mutations1 = [np.random.normal(mu, sigma) for x in
                           range(n_vars)]  # creates some random numbers between -.1 and .1
        gene_mutations2 = [np.random.normal(mu, sigma) for x in
                           range(n_vars)]
        genotype_child1 = list(keep_within_boundaries(np.add(gene_mutations1, child1), -1, 1))
        genotype_child2 = list(keep_within_boundaries(np.add(gene_mutations2, child2), -1, 1))
        newborns.append(genotype_child1)
        newborns.append(genotype_child2)  # offspring is added to the population (born).
    return newborns

def flip_coin_crossover(mom,dad):
    genotype_child = []
    # decide for each chromosome-to-be whether to take chromosome from mum or dad
    for i in range(len(mom)): 
        coin = rand.uniform(0, 1) # flip a twosided coin to decide
        # if the chromosome should be taken from moms genotype or from dads genotype
        if coin <= 0.5:
            genotype_child.append(mom[i])
        else: 
            genotype_child.append(dad[i])

    return genotype_child            


def select_survivors(population, n_deaths):  # this method kills a specified number of the least fit individual
    order_pop = ordered_population(population)
    survivors = order_pop[0:-n_deaths]  # we kill n_deaths individuals
    deaths = order_pop[-n_deaths:]# of the inverted population list: the weakest individuals
    for individual in deaths :
        print("\nstrategy " + str(individual[n_vars]) + " died")
    return survivors

def print_ordered_population_nicely(population):
    for individual in population:
        print(individual[n_vars], individual[n_vars+1]),  # print only the individual ID and the fitness

def get_statistics (population):
    pop = ordered_population(population)
    fit_list = []
    life_list = []
    for individual in pop:
        fit_list.append(individual[n_vars+1])
        life_list.append(individual[n_vars+2])
    SD_playerlife = np.std(life_list)
    Average_playerlife = np.mean(life_list)
    SD_fitness = np.std(fit_list)
    Average_fitness= np.mean(fit_list)
    Best_fitness = pop[0][n_vars+1]
    sd_weights = 0
    for i in range(n_vars):
        list_weight = []
        for individual in pop:
            list_weight.append(individual[i])
        sd_weights += np.std(list_weight) / n_vars
    return [Average_fitness, Best_fitness, sd_weights, Average_playerlife, SD_fitness, SD_playerlife]

def csv_results (run, statistics):
    if run == 1:
        csv_results = open(experiment_name + '/results.csv', 'w')
    else:
        csv_results = open(experiment_name + '/results.csv', 'a')
    writer = csv.writer(csv_results)
    writer.writerows([statistics[0]])
    writer.writerows([statistics[1]])
    writer.writerows([statistics[2]])
    writer.writerows([statistics[3]])
    writer.writerows([statistics[4]])
    writer.writerows([statistics[5]])
    csv_results.close()

def write_results(run,statistics):
    if run == 1:
        file_results = open(experiment_name + '/results.txt', 'w')
        file_results.write('Tested Enemy: ' + str(enemy) + '\n')
    else:
        file_results = open(experiment_name + '/results.txt', 'a')

    file_results.write('RUN ' + str(run) + '\n\n')
    file_results.write('# of generations: ' + str(len(statistics[0])-1) + '\n')
    file_results.write('# of fitness evaluations: ' + str(id_individual -1) + '\n')

    file_results.write('Best fitness: ' + str(statistics[1][-1]) + '\n\n')

    # we compute the average fitness of all generations, by averaging the averages... same for standard deviation...
    file_results.write(
        'Average of the fitness over the generations of this run: ' + str(np.mean(statistics[0][1:])) + '\n')
    file_results.write('Average standard deviation of the fitness over the generations of this run: ' + str(
        np.mean(statistics[4][1:])) + '\n\n')

    file_results.write('List with best fitness over the generations: \n ' + str(statistics[1]) + '\n')
    file_results.write('List with average fitness over the generations: \n' + str(statistics[0]) +'\n')
    file_results.write('List with standard deviation of the weights over the generations: \n ' + str(statistics[2]) +'\n')
    file_results.write('List with average playerlife over the generations: \n' + str(statistics[3]) + '\n')
    file_results.write('List with standard deviation of the fitness over the generations: \n' + str(statistics[4]) + '\n')
    file_results.write('List with standard deviation of the playerlife over the generations: \n' + str(statistics[5]) + '\n\n')
    file_results.close()

def write_individual(individual):

    file_results = open(experiment_name + '/individual.txt', 'w')
    for weight in individual:
        file_results.write(str(weight) + '\n')
    file_results.close()


def perform_run(difference_threshold, run):
    print('\nGeneration 1\n')
    population = spawn()
    no_improvement = 0  # amount of generations that have not been improving
    old_best_fitness = ordered_population(population)[1][n_vars+1]
    values = get_statistics(population)
    statistics = []
    statistics.append(['AF'+ str(enemy)+str(run), values[0]])
    statistics.append(['BF'+ str(enemy)+str(run), values[1]])
    statistics.append(['SW'+ str(enemy)+str(run), values[2]])
    statistics.append(['AP'+ str(enemy)+str(run), values[3]])
    statistics.append(['SF'+ str(enemy)+str(run), values[4]])
    statistics.append(['SP'+ str(enemy)+str(run), values[5]])
    generation = 0

    while no_improvement < difference_threshold:
        print('\n Generation ' + str(len(statistics[0])) + '\n')
        newborns = mate(n_randparents, population)
        n_deaths = len(newborns)
        parents_newborns = add_individuals_to_population(population, newborns)
        population = select_survivors(parents_newborns,n_deaths)
        values = get_statistics(population)
        statistics[0].append(values[0])
        statistics[1].append(values[1])
        statistics[2].append(values[2])
        statistics[3].append(values[3])
        statistics[4].append(values[4])
        statistics[5].append(values[5])
        best_fitness = ordered_population(population)[0][n_vars+1]
        print_ordered_population_nicely(population)
        best_individual = population[0][0:n_vars]
        best_individual.append(best_fitness)
        if best_fitness > old_best_fitness:
            old_best_fitness = best_fitness
            no_improvement = 0
        else:
            no_improvement += 1
        print('\nNo improvement: ' + str(no_improvement)+ '\n')
        generation += 1
    return statistics , best_individual

def main():
    best_fit = -100
    best_idd = []
    for run in range(1,n_runs+1) :
        print('RUN: ' + str(run))
        statistics, best_ind = perform_run(difference_treshold,run)
        csv_results(run,statistics)
        write_results(run, statistics)
        if best_ind[n_vars] > best_fit:
            best_fit = best_ind[n_vars]
            best_idd = best_ind[0:n_vars]
        else:
            best_idd = best_idd
            best_fit = best_fit
        global id_individual
        id_individual = 1
    write_individual(best_idd)
    print('\n Highest fitness of 10 runs is: ' + str(best_fit) + '\n')
    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')


main( )
