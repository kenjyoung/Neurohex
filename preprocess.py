import numpy as np
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
white = 0
black = 1
west = 2
east = 3
north = 4
south = 5
boardsize = 13
padding = 2
input_size = boardsize+2*padding
neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))

def neighbors(cell):
		"""
		Return list of neighbors of the passed cell.
		"""
		x = cell[0]
		y=cell[1]
		return [(n[0]+x , n[1]+y) for n in neighbor_patterns\
			if (padding<=n[0]+x and n[0]+x<boardsize+padding and padding<=n[1]+y and n[1]+y<boardsize+padding)]

def flood_fill(game, cell, edge):
	game[edge, cell[0], cell[1]] = 1
	for n in neighbors(cell):
		if(game[white, n[0], n[1]] and not game[edge, n[0], n[1]]):
			flood_fill(game, n, edge)


def new_game():
	game = np.zeros((6,input_size,input_size), dtype=bool)
	game[white, 0:padding, :] = 1
	game[white, input_size-padding:, :] = 1
	game[west, 0:padding, :] = 1
	game[east, input_size-padding:, :] = 1
	game[black, :, 0:padding] = 1
	game[black, :, input_size-padding:] = 1
	game[north, :, 0:padding] = 1
	game[south, :, input_size-padding:] = 1
	return game

def cell(move):
	x =	ord(move[0].lower())-ord('a')+padding
	y = int(move[1:])-1+padding
	return (x,y)

#cell of the mirrored move
def cell_m(move):
	x = int(move[1:])-1+padding
	y = ord(move[0].lower())-ord('a')+padding
	return (x,y)

def play_cell(game, cell, color):
	x =	cell[0]
	y = cell[1]
	edge1_connection = False
	edge2_connection = False
	game[color, x, y] = 1
	if(color == white):
		edge1 = east
		edge2 = west
	else:
		edge1 = north
		edge2 = south
	for n in neighbors((x,y)):
		if(game[edge1, n[0], n[1]]):
			east_connection = True
		if(game[edge2, n[0], n[1]]):
			west_connection = True
	if(edge1_connection):
		flood_fill(game, (x,y), edge1)
	if(edge2_connection):
		flood_fill(game, (x,y), edge2)

def preprocess(filename):
	infile = open(filename, 'r')
	positions = []
	for line in infile:
		gameW = new_game() #white plays first in this game
		gameB = new_game() #equivalent game where black plays first
		moves = line.split()
		#don't want to add terminal states to initialization positions
		del moves[-1]
		move_parity = 0
		for move in moves:
			play_cell(gameB, cell(move), white if move_parity else black)
			play_cell(gameW, cell_m(move), black if move_parity else white)
			move_parity = not move_parity
			positions.append(np.copy(gameB if move_parity else gameW))
	return positions

preprocess("raw_games.dat")

