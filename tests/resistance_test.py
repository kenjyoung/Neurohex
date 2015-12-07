import sys
sys.path.append("..")
from inputFormat import state_string, new_game
from resistance import score
from preprocess import preprocess
import numpy as np
import time
"""
Calculate resistance over 10 games and time how long it takes.
"""


pos = preprocess("../data/raw_games_small.dat", trim_final = False)

np.set_printoptions(suppress=True, precision=4, linewidth=150)

start = time.time()
for i in range(len(pos)):
	s = score(pos[i],0)
	if(np.min(np.select([s>-1],[s]))<0 and np.max(s)>0):
		print s
		print state_string(pos[i])
	#print score(pos[i], 0)
print "Computed ", len(pos), "positions in ", time.time() - start, "s"

print state_string(pos[60])
print score(pos[60], 0)