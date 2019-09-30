# 29 September 2019
# Group 17

# Goal: get data from results.csv files and make line plots of average fitness.

import csv
import numpy as np
import matplotlib.pyplot as plt

ENEMIES = [5,7] # enemies that we battle against!! [1,5,7]
ALGORITHMS = [1,2] # algorithms that we test! can be either 1 or 2 or both
MAX_NO_GENERATIONS = 50 

def boxplot_of_all_runs(enemy_id, algorithm_id):
	# makes a boxplot of the best fitness values, 1 dot in boxplot corresponds to 1 run

	directory = 'specialist_enemy' + str(enemy_id) + '_Algoritme' + str(algorithm_id) # format is 'specialist_enemy5_Algoritme1'
	lines = open(directory + '/results.txt').readlines()

	bf = []
	for line in lines: 
		if line.startswith("Best fitness:"): 
			bf_run = line.split()[2] 
			bf.append(bf_run)

	print(bf)
	bf = np.array(bf).astype(np.float)

	plt.boxplot(bf)

#	plt.savefig('boxplot_best_fitness_' + 'alg_' + str(algorithm_id) +'_en_' + str(enemy_id) + '.png')
#	plt.show()

def average_over_runs(enemy_id, algorithm_id):
	# makes a line plot of the average fitness over 10 runs against enemy_id using algorithm_id

	directory = 'specialist_enemy' + str(enemy_id) + '_algoritme' + str(algorithm_id) # format is 'specialist_enemy5_Algoritme1'
	lines = open(directory + '/results.csv').readlines()

	results = [0] * MAX_NO_GENERATIONS 
	nof_generations = [] # the number of generations differs per run, so we save them in this list

	for line in lines: # each line represents a run
		if line.startswith("AF"): #AF = average fitness, AP = average player life, SF = standard deviation fitness, SP = standard deviation
		# player life
			fitnesses = line.split(',')[1:] # list of average fitness per generation for this run
			nof_zeros_to_pad = MAX_NO_GENERATIONS - len(fitnesses) 
			nof_generations.append(len(fitnesses)) 
			fitnesses = np.pad(fitnesses, (0, nof_zeros_to_pad), 'constant') # fill list with zeros for practicality
			results = np.add(results, list(map(float,fitnesses))) # average fitnesses per generation are summed over all runs and stored in results

	results = results[~np.isnan(results)]

	final_averages = [] 

	for generation in range(1, max(nof_generations)):
		#count how many times this generation occured in 10 runs. we should divide by this amount
		denominator = [1 if x >= generation else 0 for x in nof_generations]
		nof_runs_that_contain_generation = sum(denominator)
		average_runs = results[generation-1]/nof_runs_that_contain_generation
		final_averages.append(average_runs)

	return final_averages


def main():
	# for each algorithm, for each enemy, we plot a line, that represents
	# the average fitness over the 10 runs. # all enemies come in 1 figure. 1 figure per algorithm.

	# we make 1 plot for the avg fitness of all en/alg combis
	plt.figure()
	for algorithm_id in ALGORITHMS:
		for enemy_id in ENEMIES:
			fit_avgs = average_over_runs(enemy_id, algorithm_id)
			plt.plot(fit_avgs, label= 'enemy' + str(enemy_id) + '_alg_' + str(algorithm_id))
	plt.title('Average fitness over generations')
	plt.ylabel('Average fitness over 10 runs')
	plt.xlabel('Generation')
	plt.legend()
	plt.savefig('avg_fitnes' + '.png')
	plt.show()

	# 1 figure per enemy (since enemies differ a lot, it is ugly to plot all enemies in 1 boxplot).
	# 2 algorithms per figure
	#for enemy_id in ENEMIES:
#		plt.figure()#
	#	for algorithm_id in ALGORITHMS:

	#		boxplot_of_all_runs(enemy_id, algorithm_id)

	#	plt.title("Boxplot of best fitness enemy " + str(enemy_id))
#		plt.ylabel("Fitness")#
		#plt.savefig('boxplot_best_fitness_' + 'en_' + str(enemy_id) + '.png')
		#plt.show()
		

main()
