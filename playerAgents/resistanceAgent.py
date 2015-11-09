import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from resistance import score
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput

class resistanceAgent:
	def __init__(self, state = gamestate(13)):
		self.state = copy(state)
		self.scores = None

	def move(self, move):
		"""
		Make the passed move.
		"""
		self.state.play(move)

	def search(self, time_budget):
		"""
		Compute resistance for all moves in current state.
		"""
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		raw_scores = score(stateToInput(self.state), toplay)
		score_padding = (boardsize - self.state.size)/2
		self.scores = raw_scores[score_padding:self.state.size+score_padding,score_padding:self.state.size+score_padding]

	def best_move(self):
		"""
		Return the best move according to the current tree.
		"""
		score_padding = boardsize - self.state.size
		return np.unravel_index(self.scores.argmax(), self.scores.shape)

	def set_gamestate(self, state):
		self.state = deepcopy(state)