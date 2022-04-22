import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput
import tensorflow as tf
from tensorflow.keras import models
import math

class sixInputsAgent:
	def __init__(self, state = gamestate(13)):
		self.state = copy(state)
		self.model = models.load_model('../savedModels/6Inputs')
		self.input = new_game(13)
		self.name = "sixInputs"
	def move(self, move, color):
		"""
		Make the passed move.
		"""
		self.state.play(move)
		self.input = play_cell(self.input,self.pad(move),color)
	
	def pad(self,cell):
		return (cell[0] +2,cell[1]+2)


	def openingMove(self, move, color):
		"""
		Make the passed move.
		"""
		self.input = play_cell(self.input,self.pad(move),color)

	def register(self, interface):
		interface.register_command("scores", self.gtp_scores)

	def gtp_scores(self, args):
		self.search()
		out_str = "gogui-gfx:\ndfpn\nVAR\nLABEL "
		for i in range(self.state.size*self.state.size):
			cell = np.unravel_index(i, (self.state.size,self.state.size))
			raw_cell = (cell[0]+(boardsize-self.state.size+1)/2, cell[1]+(boardsize-self.state.size+1)/2)
			toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
			if(toplay == black):
				cell = cell_m(cell)
			score_index = boardsize*raw_cell[0]+raw_cell[1]
			out_str+= chr(ord('a')+cell[0])+str(cell[1]+1)+" @"+str(self.scores[score_index])[0:6]+"@ "
		out_str+="\nTEXT scores\n"
		print(out_str)
		return(True, "")


	def search(self, time_budget = 1):
		"""
		Compute resistance for all moves in current state.
		"""
		state = self.input

		inputState = state.transpose(1,2,0)
		inputState = np.array([inputState])
		inputState = np.delete(inputState,np.s_[6:18],3)
		#get equivalent white to play game if black to play
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			state = mirror_game(state)
		played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
		self.scores = np.reshape(self.model.predict(inputState),(169))
		#set value of played cells impossibly low so they are never picked
		self.scores[played] = -math.inf
		

	def best_move(self):
		"""
		Return the best move according to the current tree.
		"""
		self.search()
		move = np.unravel_index(self.scores.argmax(), (boardsize,boardsize))
		#correct move for smaller boardsizes
		#flip returned move if black to play to get move in actual game
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			move = cell_m(move)
		print(move)
		return move

	def set_gamestate(self, state):
		self.state = deepcopy(state)

	def resetInput(self):
		self.input = new_game(13)