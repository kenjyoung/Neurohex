import numpy as np
from inputFormat import *

def get_empty(state):
	count = 0
	indices = []
	for x in range(input_size):
		for y in range(input_size):
			if(state[white,x,y] == 0 and state[black,x,y] == 0):
				count+=1
				indices.push((x,y))
	return count, indices


def fill_connect(state, cell, checked):
	checked[cell] = True
	connected = set()
	for n in neighbors(cell):
		if(state[white,n] && not checked[n]):
			fill_connect(n)
		elif(!state[black,n]):
			connected.add(n)
	return connected



def get_connected(state, cell, checked):
	connected = set()
	for n in neighbors(cell):
		if(state[white, n] && not checked[n]):
			connected |= fill_connect(state, n, checked)
		elif(!stat[black,n]):
			connected |= n
	return connected



def score(state):
	"""
	Output a resistance heuristic score over all empty nodes:
		-Treat the west edge connected nodes as one source with voltage 1
		-Treat east edge connected nodes as one source with voltage 0
		-Treat edges between empty nodes, or from source/dest to an empty node as resistors with conductance 1
		-Treat non edge connected edges to white nodes as perfect conductors
		-Treat edges to black nodes as perfect resistors
	"""
	num_empty, index_to_location = get_empty(state)
	index_to_location = index_to_location
	location_to_index = {index_to_location[i]:i for i in range(len(index_to_location))}

	#current through internal nodes except that from source and dest
	#(zero except for source and dest connected nodes)
	I = np.zeros(numempty)

	#conductance matrix such that G*V = I
	G = np.zeros(num_empty, num_empty)

	checked = np.zeros((input_size, input_size), dtype=bool)
	source_connected = fill_connect(state, (0,0), checked)
	for n in source_connected:
		j = location_to_index[n]
		I[j] -= 1
		G[j,j] += 1


	dest_connected = fill_connect(state, (input_size-1,input_size-1), checked)
	for n in dest_connected:
		j = location_to_index[n]
		G[j,j] +=1

	for i in range(len(index_to_location)):
		connected = get_connected(state, index_to_location[i], checked)
		for n in connected:
			j = location_to_index[n]
			if(j!=i):
				if(G[i,j] == 0) G[i,j] -= 1
				if(G[j,i] == 0) G[j,i] -= 1
				G[i,i] += 1

	#voltage at cells
	V = np.linalg.solve(G,I)
	#energy disipation in cell
	E = np.zeros(num_empty)
	#conductance from source to dest
	C = 0

	for i in range(num_empty):
		for j in range(num_empty):
			if(i!=j && G[i,j] != 0):
				E[i] += abs(G[i,j]*(V[j] - V[i])*(V[j] - V[i]))
			if(index_to_location[i] in source_connected):
				C += V[i] - V[j]







