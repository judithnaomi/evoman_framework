#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys,os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(),
		  		  speed="normal",
				  enemymode="static",
				  level=2)

#sol = np.loadtxt('Assignment2_Generalist_Algorithm1/individual.txt') 	# Algorithm 1
sol = np.loadtxt('competition_alg2_78/individual.txt')					# Algorithm 2
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')

# tests saved demo solutions for each enemy
for en in range(1, 9):
	# Update the number of neurons for this specific example
	env.player_controller.n_hidden = [10]
	
	#Update the enemy
	env.update_parameter('enemies',[en])

	env.play(sol)

print('\n  \n')
