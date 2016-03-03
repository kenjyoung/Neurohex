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
    	return np.unravel_index(scores.argmax(), scores.shape)
    return (np.random.choice(range(boardsize)),np.random.choice(range(boardsize)))
)


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

numEpisodes = 100
batch_size = 10

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	f.close()
else:
	print "building model..."
	network = network(batch_size=batch_size)
)

#zeros used for running network on a single state without modifying batch size
zero_inputs = theano.shared(np.zeros(np.concatenate(network.batch_size,input_shape)))
evaluate_model = theano.function(
	[inputs],
	network.output[0],
	givens={
        network.input: concatenate(input_state,zero_inputs),
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
		move_cell = policy(gameW if move_parity else gameB, evaluate_model)
		play_cell(gameB, move_cell, white if move_parity else black)
		play_cell(gameW, cell_m(move_cell), black if move_parity else white)
		move_parity = not move_parity

print "saving network..."
if args.save:
	f = file(args.save, 'wb')
else:
	f = file('mentor_network.save', 'wb')
cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()