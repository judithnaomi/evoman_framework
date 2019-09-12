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
from math import fabs,sqrt
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

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train' # train or test

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 4
gens = 30
mutation = 0.2
last_best = 0

################################### OWN PART ###########################################

n_generations = 20
n_deaths = 2 # number of individuals that dies each generation

population = {} # population is a dictionary with keys individual IDs
#and values are tuples where the first element is the list of nvars real numbers
#between -1 and 1, and the second element is the fitness value of this individual.
#e.g. population['1'] = [[-0.433434,-0.4324234,0.58532, ..., ], 0.89] here, individual
# 1 has fitness 0.89 and [-0.4333434,...] etc is its representation.




# simulation(env,x) is taken from EvoMan FrameWork - V1.0 2016 by Karine Miras
# simulates a game with game strategy x, outputs fitness according to
# default fitness function
def simulation(env,x): 
	f,p,e,t = env.play(pcont=x)
	return f

class world: # a "world" where individuals live
    def __init__(self): # the population starts as an empty list    
   
       self.population = {}
        
    def spawn(self, size):# a prespecified number (''size'') of individuals is spawned
    # and immediately added to the population with dummy fitness value of 0
        
        nof_individuals = 0

        for times in range(0,size):# these individuals are characterized by their genotype
            individual = []
            for i in range(0,n_vars):
                individual.append(rand.random()*2-1)# their genotype is a combination of n_vars random numbers between -1 and 1

            nof_individuals += 1

            self.population[nof_individuals] = (0, individual) #dummy fitness value is 0
     
      
    def fitness(self, individual):
        return simulation(env,individual) 
    
    def order_population(self):

        self.population = sorted(self.population.values(), key = 
             lambda v:(v[0]), reverse=True)   #value looks like (fitness, list of reals) so we want
        #to sort on fitness
        

    def selection(self, n_deaths):#this method kills a specified number of the least fit individuals
        # individuals die depending on their fitness, therefore we first order them
        # according to their fitness:

        world_two.order_population()

        self.population = np.array(self.population.tolist()[:-n_deaths])#the 'n_deaths'- last individuals are removed from the gene pool


    def mate (self):
        # individuals mate depending on their fitness. we first order them according
        # to their fitness:
        world_two.order_population()

        newborns = []
        for i, individual in enumerate(self.population.tolist()):
            if i%2 == 0 and i < (len(self.population.tolist())-1): #we want them to be monogamous and we do not want an error for the last one.

                offspring = [(g + h) / 2 for g, h in zip(self.population.tolist()[i], self.population.tolist()[i+1])]#the numbers of the  fittest with their subsequent individual are averaged to create a baby always skipping one (monogamous).
                gene_mutations = [(rand.random()-.5)*.1 for x in range(n_vars)] #creates some random numbers between -.1 and .1                

                offspring = [sum(x) for x in zip(gene_mutations, offspring)] # the random numbers are added to the offsprings list to create mutations in its genome.

                newborns += [offspring] #offspring is added to the population (born).            
        return newborns 
        #self.population = np.array(self.population.tolist() + newborns)

    
world_two = world()

print('Init')

world_two.spawn(npop) #for simplicity start with an even population

# for each individual, we want to compute the fitness:
fitness_values = []

for i in range (len(world_two.population)):

    individual = world_two.population[i][1] 
    fitness_of_individual = world_two.fitness(individual) #simulate game and compute their fitness
    world_two.population[i]=(fitness_of_individual, individual)


print('Spawn')
for individual in world_two.population:	
   print(individual)



for generation in range(n_generations):
    print('Generation:' + str(generation + 1))
    newborns = world_two.mate()

    for newborn in newborns:
        fitness_newborn = fitness(newborn)
        world_two.population[len(world_two.population)] = (fitness_newborn, newborn) #add newborn to population

    print('Offspring produced')

    world_two.selection(n_deaths)

    print('Selection made, individuals died')

    




	        





