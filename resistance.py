import numpy as np
from inputFormat import *
import sys

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
		if(not checked[n]):
			if(state[color,n[0], n[1]]):
				connected = connected | fill_connect(state, n, color, checked)
			elif(not state[other(color), n[0], n[1]]):
				connected.add(n)
	return connected


def get_connections(state, color, empty, checked):
	connections = {cell:set() for cell in empty}
	for cell in empty:
		for n in neighbors(cell):
			if(not checked[n]):
				if(state[color, n[0], n[1]]):
					connected_set = fill_connect(state, n, color, checked)
					for c1 in connected_set:
						for c2 in connected_set:
							connections[c1].add(c2)
							connections[c2].add(c1)
				elif(not state[other(color),n[0],n[1]]):
					connections[cell].add(n)
	return connections


def resistance(state, empty, color):
	"""
	Output a resistance heuristic score over all empty nodes:
		-Treat the west edge connected nodes as one source node with voltage 1
		-Treat east edge connected nodes as one destination node with voltage 0
		-Treat edges between empty nodes, or from source/dest to an empty node as resistors with conductance 1
		-Treat edges to white nodes (except source and dest) as perfect conductors
		-Treat edges to black nodes as perfect resistors
	"""
	if(winner(state)!=None):
		if winner(state) == color:
			return np.zeros((input_size, input_size)), float("inf")
		else:
			return np.zeros((input_size, input_size)), 0
	index_to_location = empty
	num_empty = len(empty)
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
		I[j] += 1
		G[j,j] += 1
		

	dest_connected = fill_connect(state, (input_size-1,input_size-1), color, checked)
	for n in dest_connected:
		j = location_to_index[n]
		G[j,j] +=1

	adjacency = get_connections(state, color, index_to_location, checked)
	for c1 in adjacency:
		j=location_to_index[c1]
		for c2 in adjacency[c1]:
			i=location_to_index[c2]
			G[i,j] -= 1
			G[i,i] += 1

	#voltage at each cell
	V = np.linalg.solve(G,I)

	V_board = np.zeros((input_size, input_size))
	#test code remove:
	for i in range(num_empty):
		V_board[index_to_location[i]] = V[i]

	#current passing through each cell
	Il = np.zeros((input_size, input_size))
	#conductance from source to dest
	C = 0

	for i in range(num_empty):
		if index_to_location[i] in source_connected:
			Il[index_to_location[i]] += abs(V[i] - 1)/2
		if index_to_location[i] in dest_connected:
			Il[index_to_location[i]] += abs(V[i])/2
		for j in range(num_empty):
			if(i!=j and G[i,j] != 0):
				Il[index_to_location[i]] += abs(G[i,j]*(V[i] - V[j]))/2
				if(index_to_location[i] in source_connected and
				 index_to_location[j] not in source_connected):
					C+=-G[i,j]*(V[i] - V[j])

	return Il, C


def score(state, color):
	current_credit = 2
	"""
	Score is an estimate of action value for each move, computed using the ratio
	of effective conductance for the current player and their opponent.
	The effective conductance is computed heuristically by taking the true 
	conductance and giving some credit for the current flowing through the cell 
	to be played for both the player and their oppenent.
	"""
	Q = {}
	num_empty, empty = get_empty(state)
	filled_fraction = (boardsize**2-num_empty+1)/boardsize**2
	I1, C1 = resistance(state, empty, color)
	I2, C2 = resistance(state, empty, other(color))

	num_empty, empty = get_empty(state)
	for cell in empty:
		#this makes some sense as an approximation of
		#the conductance of the next state
		C1_prime = C1 + I1[cell]**2/(3*(1-I1[cell]))
		C2_prime = max(0,C2 - I2[cell])
		if(C1_prime>C2_prime):
			Q[cell] = 1 - C2_prime/C1_prime
		else:
			Q[cell] = C1_prime/C2_prime - 1

	output = -1*np.ones((boardsize, boardsize))
	for cell, value in Q.iteritems():
		output[cell[0]-padding, cell[1]-padding] = value
	return output 
