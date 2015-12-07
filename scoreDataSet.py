import numpy as np
from resistance import score
from preprocess import *

positions = preprocess("data/raw_games.dat")
print "scoring positions..."
scores = np.empty((positions.shape[0],boardsize,boardsize))
num_positions = positions.shape[0]
output_interval = num_positions/10
for i in range(num_positions):
	if(i%output_interval == 0):
		print "completion: ",i/output_interval
	try:
		scores[i]=score(positions[i], 0)
	except np.linalg.linalg.LinAlgError:
		print "singular position at ",str(i),": " state_string(positions[i])
		score[i]=-1*np.ones((boardsize, boardsize))

print "saving to file..."
savefile = open("data/scoredPositions.npz", 'w')
np.savez(savefile, positions=positions, scores=scores)