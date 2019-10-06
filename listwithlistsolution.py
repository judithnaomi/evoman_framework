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
import csv

experiment_name = 'test'  # Assignment task 1
enemy = 2 # Set correct enemy to run in this solution
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
n_runs = 2
n_hidden = 10
n_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 10 hidden neurons
npop = 10
n_deaths = int(npop / 5) * 2
id_individual = 1
difference_treshold = 2

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
        fitness, player_life = simulation(individual[0:n_vars])
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


def mate(variation,percentage, population):
    newborns = []
    pop = ordered_population(population)
    random_numbers = list(range(0, len(pop)) ) # generates a list counting from 0 up to n_pop
    rand.shuffle(random_numbers)  # shuffle the numbers

    for i in range(0, len(pop), 5):  # in steps of 5
        five_random = [random_numbers[i], random_numbers[i + 1], random_numbers[i + 2], random_numbers[i + 3],
                       random_numbers[i + 4]]  # makes a list with 5 random numbers
        five_random.sort()  # sorts these number from low to high

        # the first mating pair; chosing the lowest random number automatically gives the one with the highest fitness becaus pop is ordered on fitness
        dad1 = pop[five_random[0]]
        mom1 = pop[five_random[1]]

        # the second mating pair
        dad2 = pop[five_random[2]]
        mom2 = pop[five_random[3]]

        # creating the children
        genotype_child1 = ((np.array(mom1[0:n_vars]) * percentage) + (np.array(dad1[0:n_vars]))* (1-percentage))
        genotype_child2 = (np.array(mom2[0:n_vars]) + np.array(dad2[0:n_vars]))/2

        gene_mutations1 = [rand.uniform(-variation, variation) for x in
                           range(n_vars)]  # creates some random numbers between -.1 and .1
        gene_mutations2 = [rand.uniform(-variation, variation) for x in
                           range(n_vars)]

        genotype_child1 = list(keep_within_boundaries(np.add(gene_mutations1, genotype_child1), -1, 1))
        genotype_child2 = list(keep_within_boundaries(np.add(gene_mutations2, genotype_child2), -1, 1))


        # append the children to the newborns
        newborns.append(genotype_child1)
        newborns.append(genotype_child2)  # offspring is added to the population (born).
    return newborns

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

def perform_run(upsilon,percentage, difference_threshold, run):
    print('\nGeneration 1\n')
    population = spawn()
    no_improvement = 0  # amount of generations that have not been improving
    old_best_fitness = ordered_population(population)[0][n_vars+1]
    values = get_statistics(population)
    statistics = []
    statistics.append(['AF'+ str(enemy)+str(run), values[0]])
    statistics.append(['BF'+ str(enemy)+str(run), values[1]])
    statistics.append(['SW'+ str(enemy)+str(run), values[2]])
    statistics.append(['AP'+ str(enemy)+str(run), values[3]])
    statistics.append(['SF'+ str(enemy)+str(run), values[4]])
    statistics.append(['SP'+ str(enemy)+str(run), values[5]])

    while no_improvement < difference_threshold:
        print('\n Generation ' + str(len(statistics[0])) + '\n')
        newborns = mate(upsilon, percentage, population)
        parents_newborns = add_individuals_to_population(population, newborns)
        population = select_survivors(parents_newborns,n_deaths)
        values = get_statistics(population)
        statistics[0].append(values[0])
        statistics[1].append(values[1])
        statistics[2].append(values[2])
        statistics[3].append(values[3])
        statistics[4].append(values[4])
        statistics[5].append(values[5])
        best_fitness = values[1]
        print_ordered_population_nicely(population)
        if best_fitness > old_best_fitness:
            old_best_fitness = best_fitness
            no_improvement = 0
        else:
            no_improvement += 1
        print('\nNo improvement: ' + str(no_improvement)+ '\n')
    return statistics

def main(upsilon, ratio):
    for run in range(1,n_runs+1) :
        print('RUN: ' + str(run))
        statistics = perform_run(upsilon,ratio,difference_treshold,run)
        csv_results(run,statistics)
        write_results(run, statistics)
        global id_individual
        id_individual = 1

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')


main(0.1, 0.5)
