import theano
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import cPickle
import argparse
from copy import copy


def data_shuffle(x, y):
	rng_state = rand.get_state()
	rand.shuffle(x)
	rand.set_state(rng_state)
	rand.shuffle(y)

def policy(state, evaluator):
	rand = np.random.random()
    if(rand>Emu):
    	scores = evaluator(state)
    	return scores.argmax()
    return (np.random.choice(range(boardsize)),np.random.choice(range(boardsize)))

def Q_update():
	batch = np.random.choice(np.arange(0,replay_capacity), size=(batch_size))
	scores1 = evaluate_model_batch(state1_memory[batch])
	scores2 = evaluate_model_batch(state2_memory[batch])
	actions = action_memory[batch]
	rewards = reward_memory[batch]
	#ToDo: finish this************
	






parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
args = parser.parse_args()

print "loading starting positions... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

input_state = T.tensor3('input_state')

state_batch = T.tensor4('state_batch')
value_batch = T.fvector('value_batch')
action_batch = T.fmatrix('action_batch')


replay_capacity = 100000

#replay memory from which updates are drawn
replay_index = 0
replay_full = False
state1_memory = np.zeros(np.concatenate(replay_capacity, input_shape), dtype=bool_)
action_memory = np.zeros(replay_capacity, dtype=int8)
reward_memory = np.zeros(replay_capacity, dtype=bool_)
state2_memory = np.zeros(np.concatenate(replay_capacity, input_shape), dtype=bool_)

numEpisodes = 100
batch_size = 10

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
)

#zeros used for running network on a single state without modifying batch size
zero_inputs = theano.shared(np.zeros(np.concatenate(network.batch_size,input_shape)))
evaluate_model_single = theano.function(
	[input_state],
	network.output[0],
	givens={
        network.input: concatenate(input_state,zero_inputs),
	}
)

evaluate_model_batch = theano.function(
	[input_batch],
	network.output,
	givens={
        network.input: input_state,
	}
)

cost = T.mean(T.sqr(network.output[T.arange(y.shape[0]),action] - y))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
	[state_batch],
	[value_batch],
	[action_batch],
	updates = upgates,
	givens={
		network.input: state_batch,
		action: action_batch
		y: value_batch
	}
)

for i in range(numEpisodes):
	#randomly choose who is to move from each position to increase variability in dataset
	move_parity = np.randint(2)
	#randomly choose starting position from database
	index = np.randint(numPositions)
	gameW = copy(positions(index))
	gameB = mirror_game(gameW)
	while(winner(gameW)==None):
		action = policy(gameW if move_parity else gameB, evaluate_model_single)
		action_memory[replay_index] = action
		state1_memory[replay_index,:,:] = copy(gameW if move_parity else gameB)
		move_cell = np.unravel_index(action, (boardsize,boardsize) )
		play_cell(gameB, move_cell, white if move_parity else black)
		play_cell(gameW, cell_m(move_cell), black if move_parity else white)
		if(not winner(gameW)==None):
			#only the player who just moved can win, so if anyone wins the reward is 1
			#for the current player
			reward_memory[replay_index] = 1
		else
			reward_memory[replay_index] = 0

		state2_memory[replay_index,:,:] = copy(gameB if move_parity else gameW)
		move_parity = not move_parity
		replay_index+=1
		if(replay_index>=replay_capacity):
			replay_full = True
			replay_index = 0

		if(replay_full):
			Q_update()





print "saving network..."
if args.save:
	f = file(args.save, 'wb')
else:
	f = file('Q_network.save', 'wb')
cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()