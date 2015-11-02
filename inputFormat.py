import numpy as np

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