import numpy as np
from resistance import score
from preprocess import *

positions = preprocess("data/raw_games.dat")
print "scoring positions..."
scores = np.empty((positions.shape[0],boardsize,boardsize))
num_positions = positions.shape[0]
output_interval = num_positions/10
for i in range(num_postions):
	if(i%output_interval == 0):
		print "completion: ",i/output_interval
	scores[i]=score(positions[i], 0)

print "saving to file..."
savefile = open("data/scoredPositions.npz", 'w')
np.savez(savefile, positions=positions, scores=scores)