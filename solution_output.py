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

experiment_name = 'assignment_specialist1'  # Assignment task 1
enemy = 2  # Set correct enemy to run in this solution
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

population = {}  # population is a dictionary with keys individual IDs
# and values are tuples where the first element is the fitness value and
# the second element is the list neutral network weights between -1 and 1.
# e.g. population['1'] = (0.89, [-0.433434,-0.4324234,0.58532, ..., ], mutation_sigma) here, individual
# 1 has fitness 0.89 and [-0.4333434,...] are the neural network weights.

id_individual = 0  # the id of the newest individual


def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
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


def spawn(size):  # a prespecified number (''size'') of individuals is spawned
    initial_population = []
    for times in range(0, size):  # these individuals are characterized by their genotype
        individual = []
        for i in range(0, n_vars):
            individual.append(
                rand.random() * 2 - 1)  # their genotype is a combination of n_vars random numbers between -1 and 1

        initial_population.append(individual)

    return np.array(initial_population)


def fitness_playerlife(individual):
    return simulation(env, individual)


def ordered_population(population):
    # returns the population as a list of tuples i.e.
    # [(ID,[fitness,list_of_values]), (ID,[fitness,list_of_values]),....]
    # note: fitness in decreasing order, so starting with fittest individual
    return sorted(population.items(), key=lambda kv: kv[1][0], reverse=True)
    # e.g. kv[1]=(fitness,individual). kv[1][0] is the fitness and fitness is used to sort the list.


def mate(n_pop):
    newborns = []
    global population
    pop = ordered_population(population)
    random_numbers = list(range(0, n_pop) ) # generates a list counting from 0 up to and including n_pop -1
    rand.shuffle(random_numbers)  # shuffle the numbers

    for i in range(0, len(pop), 5):  # in steps of 5
        five_random = [random_numbers[i], random_numbers[i + 1], random_numbers[i + 2], random_numbers[i + 3],
                       random_numbers[i + 4]]  # makes a list with 5 random numbers
        five_random.sort()  # sorts these number from low to high

        # the first mating pair; chosing the lowest random number automatically gives the one with the highest fitness becaus pop is ordered on fitness
        key_dad1, value_dad1 = pop[five_random[0]]
        key_mom1, value_mom1 = pop[five_random[1]]

        # the second mating pair
        key_dad2, value_dad2 = pop[five_random[2]]
        key_mom2, value_mom2 = pop[five_random[3]]

        # creating the children
        genotype_child1 = (value_mom1[2] + value_dad1[2]) / 2
        genotype_child2 = (value_mom2[2] + value_dad2[2]) / 2

        gene_mutations1 = [(rand.random() - .5) * .1 for x in
                           range(n_vars)]  # creates some random numbers between -.1 and .1
        gene_mutations2 = [(rand.random() - .5) * .1 for x in
                           range(n_vars)]

        genotype_child1 = keep_within_boundaries(np.add(gene_mutations1, genotype_child1), -1, 1)
        genotype_child2 = keep_within_boundaries(np.add(gene_mutations2, genotype_child2), -1, 1)

        # append the children to the newborns
        newborns.append(genotype_child1)
        newborns.append(genotype_child2)  # offspring is added to the population (born).
    return newborns


def select_survivors(n_deaths):  # this method kills a specified number of the least fit individual
    global population # number of individuals that dies each generation
    deaths = ordered_population(population)[::-1][:n_deaths]  # we take the first n_deaths elements
    # of the inverted population list: the weakest individuals

    for key, value in deaths:
        del population[key]
        print("strategy " + str(key) + " died")


def add_individuals_to_population(new_individuals):
    for individual in new_individuals:
        global id_individual, population
        fitness, player_life = fitness_playerlife(individual)
        population[id_individual] = [fitness, player_life, individual]
        id_individual = id_individual + 1
        print("individual " + str(id_individual) + " added")


def print_ordered_population_nicely():
    for key, value in ordered_population(population):
        print(key, value[0]),  # print only the individual ID and the fitness

def get_average_fitness(population1):
    sum_fitness = 0
    population = ordered_population(population1)
    for i in range(len(population)):
        key1, value1 = population[i]
        sum_fitness += value1[0]
    average_fitness = sum_fitness/len(population)


    return average_fitness

def get_sd_fitness(population):
    order_pop = ordered_population(population)
    sd_list = []
    for i in range(len(order_pop)):
        key, value = order_pop[i]
        sd_list.append(value[0])
    sd_fitness = np.std(sd_list)
    return sd_fitness

def get_sd_generation(population):
    sd_generation = 0
    order_population = ordered_population(population)
    for i in range(n_vars):
        list_weight = []
        for z in range(len(order_population)):
            key, value = order_population[z]
            list_weight.append(value[2][i])
        sd_generation += np.std(list_weight)/n_vars
    return sd_generation

def get_average_playerlife(population):
    sum_playerlife = 0
    order_population = ordered_population(population)
    for i in range(len(order_population)):
        key, value = order_population[i]
        sum_playerlife += value[1]
    average_playerlife = sum_playerlife/len(order_population)

    return average_playerlife



def write_results(run, generations, average_fit , best_fitness , standard_deviation, playerlife, sd_fitness):
    if run == 1:
        file_results = open(experiment_name + '/results.txt', 'w')
        file_results.write('Tested Enemy: ' + str(enemy) + '\n')
    else:
        file_results = open(experiment_name + '/results.txt', 'a')

    file_results.write('RUN ' + str(run) + '\n\n')
    file_results.write('# of generations: ' + str(len(generations)) + '\n')
    file_results.write('# of fitness evaluations: ' + str(id_individual) + '\n')

    file_results.write('Best fitness: ' + str(ordered_population(generations[-1])[0][1][0]) + '\n\n')

    # we compute the average fitness of all generations, by averaging the averages... same for standard deviation...
    file_results.write(
        'Average of the fitness over the generations of this run: ' + str(np.mean(average_fit[1:])) + '\n')
    file_results.write('Average standard deviation of the fitness over the generations of this run: ' + str(
        np.mean(sd_fitness[1:])) + '\n\n')

    file_results.write('List with best fitness over the generations: \n ' + str(best_fitness) + '\n')
    file_results.write('List with average fitness over the generations: \n' + str(average_fit) +'\n')
    file_results.write('List with standard deviation of the weights over the generations: \n ' + str(standard_deviation) +'\n')
    file_results.write('List with average playerlife over the generations: \n' + str(playerlife) + '\n')
    file_results.write('List with standard deviation of the fitness over the generations: \n' + str(sd_fitness) + '\n\n')
    file_results.close()

def csv_results (run, average_fit , best_fitness , standard_deviation, playerlife, sd_fitness):
    if run == 1:
        csv_results = open(experiment_name + '/results.csv', 'w')
    else:
        csv_results = open(experiment_name + '/results.csv', 'a')
    writer = csv.writer(csv_results)
    writer.writerows([average_fit])
    writer.writerows([best_fitness])
    writer.writerows([standard_deviation])
    writer.writerows([playerlife])
    writer.writerows([sd_fitness])
    csv_results.close()





def perform_run(n_pop, difference_threshold, run, enemy):
    print('Generation 1')

    pioneers = spawn(n_pop)
    add_individuals_to_population(pioneers)
    generations = []
    generations.append(population.copy())
    no_improvement = 0  # amount of generations that have not been improving

    old_best_fitness = ordered_population(population)[0][1][0]

   # old_average_fitness = get_average_fitness(population)
    n_deaths = int(n_pop / 5) * 2
    list_average_fitness = ['AF'+ str(enemy)+str(run)]
    list_average_fitness.append(get_average_fitness(population))
    list_best_fitness = ['BF' + str(enemy)+str(run)]
    list_best_fitness.append(old_best_fitness)
    list_sd_weights = ['SW' + str(enemy)+str(run)]
    list_sd_weights.append(get_sd_generation(population))
    list_average_playerlife = ['AP'+ str(enemy) +str(run)]
    list_average_playerlife.append(get_average_playerlife(population))
    list_sd_fitness = ['SF'+ str(enemy) +str(run)]
    list_sd_fitness.append(get_sd_fitness(population))

    # if there is no significant improvement after .. generations, then terminate
    while no_improvement < difference_threshold:

        newborns = mate(n_pop)
        add_individuals_to_population(newborns)
        select_survivors(n_deaths)
        generations.append(population.copy())
        print_ordered_population_nicely()

        best_fitness = ordered_population(population)[0][1][0]
        #average_fitness = get_average_fitness(population)
        if best_fitness > old_best_fitness:
            old_best_fitness = best_fitness
            no_improvement = 0
        else:
            no_improvement += 1
        print('\nNo improvement: ' + str(no_improvement))
        list_average_fitness.append(get_average_fitness(population))
        list_best_fitness.append(ordered_population(population)[0][1][0])
        list_sd_weights.append(get_sd_generation(population))
        list_average_playerlife.append(get_average_playerlife(population))
        list_sd_fitness.append(get_sd_fitness(population))




    return generations, list_average_fitness, list_best_fitness, list_sd_weights, list_average_playerlife, list_sd_fitness


def main(n_pop, difference_threshold, n_runs):
    for run in range(1,n_runs+1):
        print('RUN: ' + str(run))

        generations, average_fitness, best_fitness, sd_weights, average_playerlife, sd_fitness = perform_run(n_pop, difference_threshold, run, enemy)

        write_results(run, generations, average_fitness, best_fitness , sd_weights, average_playerlife, sd_fitness)
        csv_results(run, average_fitness, best_fitness, sd_weights, average_playerlife,sd_fitness)

        global population, id_individual
        population = {}
        id_individual = 0

        print (average_fitness)
        print('\n')
        print (best_fitness)
        print('\n')
        print (sd_weights)
        print('\n')
        print(average_playerlife)
        print('\n')
        print(sd_fitness)


    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')


main(5, 3, 2)