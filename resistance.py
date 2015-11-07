import numpy as np
from inputFormat import *

def get_empty(state):
	count = 0
	indices = []
	for x in range(input_size):
		for y in range(input_size):
			if(state[white,x,y] == 0 and state[black,x,y] == 0):
				count+=1
				indices.append((x,y))
	return count, indices


def fill_connect(state, cell, color, checked):
	checked[cell] = True
	connected = set()
	for n in neighbors(cell):
		if(state[color,n[0], n[1]] and not checked[n]):
			fill_connect(state, n, color, checked)
		elif(not state[other(color), n[0], n[1]]):
			connected.add(n)
	return connected



def get_connected(state, cell, color, checked):
	connected = set()
	for n in neighbors(cell):
		if(state[color, n[0], n[1]] and not checked[n]):
			connected = connected | fill_connect(state, n, color, checked)
		elif(not state[other(color),n[0],n[1]]):
			connected.add(n)
	return connected



def resistance(state, color):
	"""
	Output a resistance heuristic score over all empty nodes:
		-Treat the west edge connected nodes as one source node with voltage 1
		-Treat east edge connected nodes as one destination node with voltage 0
		-Treat edges between empty nodes, or from source/dest to an empty node as resistors with conductance 1
		-Treat edges to white nodes (except source and dest) as perfect conductors
		-Treat edges to black nodes as perfect resistors
	"""
	num_empty, index_to_location = get_empty(state)
	index_to_location = index_to_location
	location_to_index = {index_to_location[i]:i for i in range(len(index_to_location))}

	#current through internal nodes except that from source and dest
	#(zero except for source and dest connected nodes)
	I = np.zeros(num_empty)

	#conductance matrix such that G*V = I
	G = np.zeros((num_empty, num_empty))

	checked = np.zeros((input_size, input_size), dtype=bool)
	source_connected = fill_connect(state, (0,0), color, checked)
	for n in source_connected:
		j = location_to_index[n]
		I[j] -= 1
		G[j,j] += 1


	dest_connected = fill_connect(state, (input_size-1,input_size-1), color, checked)
	for n in dest_connected:
		j = location_to_index[n]
		G[j,j] +=1

	for i in range(len(index_to_location)):
		connected = get_connected(state, index_to_location[i], color, checked)
		for n in connected:
			j = location_to_index[n]
			if(j!=i):
				G[i,j] -= 1
				G[j,i] -= 1
				G[i,i] += 1

	#voltage at each cell
	V = np.linalg.solve(G,I)
	#energy disipation in cell
	E = np.zeros(num_empty)
	#conductance from source to dest
	C = 0

	for i in range(num_empty):
		for j in range(num_empty):
			if(i!=j and G[i,j] != 0):
				E[i] += abs(G[i,j]*(V[j] - V[i])*(V[j] - V[i]))
			if(index_to_location[i] in source_connected):
				C += V[i] - V[j]

	return E, C


def score(state, color):
	temp = np.copy(state)
	num_empty, empty = get_empty(temp)
	Q = {}
	for move in empty:
		play_cell(temp, move, color)
		E1, C1 = resistance(temp, color)
		E2, C2 = resistance(temp, other(color))
		if(C1>C2):
			Q[move] = 1 - C2/C1
		else:
			Q[move] = C1/C2 - 1
		clear_cell(temp, move)

	output = np.zeros((boardsize, boardsize))
	for cell, value in Q.iteritems():
		output[cell[0]-padding, cell[1]-padding] = value
	return output 










