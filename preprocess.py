import numpy as np
from inputFormat import *
"""
raw_games.dat contains a list of moves corresponding to a 13x13 hex game on each line.
This program converts this into an array of hex positions represented as a 4x17x17
boolean tensor where the 6 channels represent (in order) white, black, west edge connection,
east edge connection, opponent north edge connection and opponent south edge
connection. The additional width (17x17 instead of 13x13) is to include the edges
of the board which are automatically colored for the appropriate player and edge 
connected appropriately, additionally this allows a 5x5 input layer filter to be
placed on directly on the edges without going out of range. Every position 
is transformed to be white-to-play (by reflection and white-black swap of 
black-to-play positions). 
"""

def preprocess(filename, trim_final = True):
	infile = open(filename, 'r')
	positions = []
	for line in infile:
		gameW = new_game() #white plays first in this game
		gameB = new_game() #equivalent game where black plays first
		moves = line.split()
		if trim_final:
			#don't want to add terminal states to initialization positions
			del moves[-1]
		move_parity = 0
		for move in moves:
			move_cell = cell(move)
			play_cell(gameB, move_cell, white if move_parity else black)
			play_cell(gameW, cell_m(move_cell), black if move_parity else white)
			move_parity = not move_parity
			positions.append(np.copy(gameB if move_parity else gameW))
		num_positions = len(positions)
		positions_array = np.empty((num_positions,6,input_size,input_size), dtype=bool)
		for i in range(num_positions):
			positions_array[i]=positions[i]
	return positions_array
