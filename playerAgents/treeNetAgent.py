import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput
import time
import theano
from theano import tensor as T
import cPickle
from math import sqrt, log

EXPLORATION = 0.1

class node:
	"""
	Node for the tree search. Stores the move applied to reach this node from its parent,
	stats for the associated game position, children, parent and outcome 
	(outcome==none unless the position ends the game).
	"""
	def __init__(self, Q = None, N = None,  move = None, parent = None):
		"""
		Initialize a new node with optional move and parent and initially empty
		children list and rollout statistics and unspecified outcome.
		"""
		assert((Q is not None or N is not None) == (Q is not None and N is not None))
		self.move = move
		self.parent = parent
		if N is not None:
			self.N = N #times this position was visited
			self.Q = Q #sum of all values encountered from this position
		else:
			self.N = 0 #times this position was visited
			self.Q = 0 #sum of all values encountered from this position
		self.children = []
		self.outcome = gamestate.PLAYERS["none"]

	def add_children(self, children):
		"""
		Add a list of nodes to the children of this node.
		"""
		self.children += children

	def set_outcome(self, outcome):
		"""
		Set the outcome of this node (i.e. if we decide the node is the end of
		the game)
		"""
		self.outcome = outcome

	def value(self, explore = 0):
		"""
		Calculate the UCT value of this node relative to its parent, the parameter
		"explore" specifies how much the value should favor nodes that have
		yet to be thoroughly explored versus nodes that seem to have a high win
		rate. 
		Currently explore is set to zero when choosing the best move to play so
		that the move with the highest winrate is always chossen. When searching
		explore is set to EXPLORATION specified above.
		"""
		#unless explore is set to zero, maximally favor unexplored nodes
		if(self.N == 0):
			if(explore == 0):
				return 0
			else:
				return inf
		else:
			return self.Q/self.N + explore*sqrt(2*log(self.parent.N)/self.N)

class treeNetAgent:
	def __init__(self, state = gamestate(13)):
		self.state = copy(state)
		self.root = node()
		f = file(os.path.dirname(os.path.realpath(__file__))+"/network.save", 'rb')
		network = cPickle.load(f)
		f.close()

		input_state = T.tensor3('input_state')
		self.evaluator = theano.function(
			[input_state],
			network.output[0],
			givens={
		        network.input: input_state.dimshuffle('x', 0, 1, 2),
			}
		)

	def move(self, move):
		"""
		Make the passed move.
		"""
		for child in self.root.children:
			#make the child associated with the move the new root
			if move == child.move:
				child.parent = None
				self.root = child
				self.state.play(child.move)
				return

		#if for whatever reason the move is not in the children of
		#the root just throw out the tree and start over
		self.state.play(move)
		self.root = node()

	def register(self, interface):
		interface.register_command("scores", self.gtp_scores)

	def gtp_scores(self, args):
		self.search(10)
		out_str = "gogui-gfx:\ndfpn\nVAR\nLABEL "
		for node in self.root.children:
			cell = np.unravel_index(node.move, (boardsize,boardsize))
			out_str+= chr(ord('a')+cell[0])+str(cell[1]+1)+" @"+str(node.value(0))[0:6]+"@ "
		out_str+="\nTEXT scores\n"
		print(out_str)
		return(True, "")

	def evaluate(self, parent, state):
		#get equivalent white to play game if black to play
		children = []
		toplay = white if state.toplay == self.state.PLAYERS["white"] else black
		state = stateToInput(state)
		if(toplay == black):
			state = mirror_game(state)
		played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
		scores = self.evaluator(state)
		value = -np.max(scores[np.logical_not(played)])
		for i in range(len(scores)):
			if not played[i]:
				if(toplay == white):
					move = i
				else:
					move = boardsize*(i%boardsize)+i//boardsize
				children.append(node(Q = scores[i], N = 1, move = move, parent = parent))
		parent.add_children(children)
		return value


	def search(self, time_budget = 1):
		"""
		Compute resistance for all moves in current state.
		"""
		startTime = time.time()
		num_evals = 0

		#do until we exceed our time budget
		while(time.time() - startTime < time_budget):
			node, state = self.select_node()
			value = self.evaluate(node, state)
			turn = state.turn()
			self.backup(node, turn, value)
			num_evals += 1
		sys.stderr.write("Ran "+str(num_evals)+ " evaluations in " +\
			str(time.time() - startTime)+" sec\n")

	def select_node(self):
		"""
		Select a node in the tree to preform an evaluation from.
		"""
		node = self.root
		state = deepcopy(self.state)

		#stop if we reach a leaf node
		while(len(node.children)!=0):
			#decend to the maximum value node, break ties at random
			node = max(node.children, key = lambda n: n.value(EXPLORATION))
			move = np.unravel_index(node.move, (boardsize, boardsize))
			#correct move for smaller boardsizes
			move = (move[0]-(boardsize-self.state.size+1)/2, move[1]-(boardsize-self.state.size+1)/2)
			state.play(move)
		return (node, state)

	def backup(self, node, turn, value):
		"""
		Update the node statistics on the path from the passed node to root to reflect
		the outcome of a randomly simulated playout.
		"""
		while node!=None:
			node.N += 1
			node.Q +=value
			value = -value
			node = node.parent

	def best_move(self):
		"""
		Return the best move according to the current tree.
		"""
		if(self.state.winner() != gamestate.PLAYERS["none"]):
			return gamestate.GAMEOVER

		#choose the move of the highest value node
		move = np.unravel_index(max(self.root.children, key = lambda n: n.value(0)).move, (boardsize, boardsize))
		#correct move for smaller boardsizes
		move = (move[0]-(boardsize-self.state.size+1)/2, move[1]-(boardsize-self.state.size+1)/2)
		return move

	def set_gamestate(self, state):
		self.state = deepcopy(state)
		self.root = node()