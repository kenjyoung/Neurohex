import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput
import theano
from theano import tensor as T
import cPickle

class networkAgent:
	def __init__(self, state = gamestate(13)):
		self.state = copy(state)
		f = file(os.path.dirname(os.path.realpath(__file__))+"/network.save", 'rb')
		network = cPickle.load(f)
		batch_size = network.batch_size
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
		self.state.play(move)

	def search(self, time_budget):
		"""
		Compute resistance for all moves in current state.
		"""
		state = stateToInput(self.state)
		#get equivalent white to play game if black to play
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			state = mirror_game(state)

		played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
		self.scores = self.evaluator(state)
		#set value of played cells impossibly low so they are never picked
		self.scores[played] = -2

	def best_move(self):
		"""
		Return the best move according to the current tree.
		"""
		move = np.unravel_index(self.scores.argmax(), (boardsize,boardsize))
		#flip returned move if black to play to get move in actual game
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			move = cell_m(move)
		return move

	def set_gamestate(self, state):
		self.state = deepcopy(state)