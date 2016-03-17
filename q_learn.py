import theano
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import cPickle
import argparse
import time
import os.path

def save():
	print "saving network..."
	if args.save:
		f = file(args.save, 'wb')
	else:
		f = file('Q_training.save', 'wb')
	cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


def epsilon_greedy_policy(state, evaluator):
	rand = np.random.random()
	played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
	if(rand>epsilon_q):
		scores = evaluator(state)
		#set value of played cells impossibly low so they are never picked
		scores[played] = -2
		#np.set_printoptions(precision=3, linewidth=100)
		#print scores.max()
		return scores.argmax()
	#choose random open cell
	return np.random.choice(np.arange(boardsize*boardsize)[np.logical_not(played)])

def softmax(x, t):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/t)
    return e_x / e_x.sum()

def softmax_policy(state, evaluator, temperature=1):
	rand = np.random.random()
	not_played = np.logical_not(np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding])).flatten()
	scores = evaluator(state)
	prob = softmax(scores[not_played], temperature)
	tot = 0
	choice = None
	for i in range(prob.size):
		tot += prob[i]
		if(tot>rand):
			choice = i
			break
	return not_played.nonzero()[0][choice]


def Q_update():
	states1, actions, rewards, states2 = mem.sample_batch(batch_size)
	scores = evaluate_model_batch(states2)
	played = np.logical_or(states2[:,white,padding:boardsize+padding,padding:boardsize+padding],\
		     states2[:,black,padding:boardsize+padding,padding:boardsize+padding]).reshape(scores.shape)
	#set value of played cells impossibly low so they are never picked
	scores[played] = -2
	targets = np.zeros(rewards.size).astype(theano.config.floatX)
	targets[rewards==1] = 1
	targets[rewards==0] = -np.amax(scores, axis=1)[rewards==0]
	cost = train_model(states1,targets,actions)
	return cost

def action_to_cell(action):
	cell = np.unravel_index(action, (boardsize,boardsize))
	return(cell[0]+padding, cell[1]+padding)

def flip_action(action):
	return boardsize*boardsize-1-action

class replay_memory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.size = 0
		self.index = 0
		self.full = False
		self.state1_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)
		self.action_memory = np.zeros(capacity, dtype=np.uint8)
		self.reward_memory = np.zeros(capacity, dtype=bool)
		self.state2_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)

	def add_entry(self, state1, action, reward, state2):
		self.state1_memory[self.index, :, :] = state1
		self.state2_memory[self.index, :, :] = state2
		self.action_memory[self.index] = action
		self.reward_memory[self.index] = reward
		self.index += 1
		if(self.index>=self.capacity):
			self.full = True
			self.index = 0
		if not self.full:
			self.size += 1

	def sample_batch(self, size):
		batch = np.random.choice(np.arange(0,self.size), size=size)
		states1 = self.state1_memory[batch]
		states2 = self.state2_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		return (states1, actions, rewards, states2)


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
#unused
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

print "loading starting positions... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

input_state = T.tensor3('input_state')

state_batch = T.tensor4('state_batch')
target_batch = T.dvector('target_batch')
action_batch = T.ivector('action_batch')


replay_capacity = 100000

#replay memory from which updates are drawn
mem = replay_memory(replay_capacity)

numEpisodes = 100000
batch_size = 64

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	batch_size = network.batch_size
	f.close()
else:
	print "building model..."
	network = network(batch_size=batch_size)
	print "network size: "+str(network.mem_size.eval())

#zeros used for running network on a single state without modifying batch size
input_padding = theano.shared(np.zeros(np.concatenate(([network.batch_size],input_shape))).astype(theano.config.floatX))
evaluate_model_single = theano.function(
	[input_state],
	network.output[0],
	givens={
        network.input: T.set_subtensor(input_padding[0,:,:,:], input_state),
	}
)

evaluate_model_batch = theano.function(
	[state_batch],
	network.output,
	givens={
        network.input: state_batch,
	}
)

cost = T.mean(T.sqr(network.output[T.arange(target_batch.shape[0]),action_batch] - target_batch))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
	[state_batch,target_batch,action_batch],
	cost,
	updates = updates,
	givens={
		network.input: state_batch,
	}
)

print "Running episodes..."
epsilon_q=0.1
try:
	for i in range(numEpisodes):
		cost = 0
		run_time = 0
		num_step = 0
		#randomly choose who is to move from each position to increase variability in dataset
		move_parity = np.random.choice([True,False])
		#randomly choose starting position from database
		index = np.random.randint(numPositions)
		gameW = positions[index]
		gameB = mirror_game(gameW)
		while(winner(gameW)==None):
			t = time.clock()
			action = epsilon_greedy_policy(gameW if move_parity else gameB, evaluate_model_single)
			#randomly flip states to capture symmetry
			if(np.random.choice([True,False])):
				state1 = gameW if move_parity else gameB
			else:
				state1 = flip_game(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
			play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
			play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
			if(not winner(gameW)==None):
				#only the player who just moved can win, so if anyone wins the reward is 1
				#for the current player
				reward = 1
			else:
				reward = 0
			#randomly flip states to capture symmetry
			if(np.random.choice([True,False]) == True):
				state2 = gameB if move_parity else gameW
			else:
				state2 = flip_game(gameB if move_parity else gameW)
			move_parity = not move_parity
			mem.add_entry(state1, action, reward, state2)
			if(mem.size > batch_size):
				cost += Q_update()
				#print state_string(gameW)
			num_step += 1
		print "Episode", i, "complete, cost: ", cost/num_step, " Time per move: ", run_time/num_step
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save()
	exit(1)

save()