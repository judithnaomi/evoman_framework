import numpy as np

f = open("individual.txt", "r")
weights = f.readlines()

weights = [float(x) for x in weights]
weights = np.array(weights)

np.savetxt('17.txt',weights)