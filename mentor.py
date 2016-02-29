import theano
import time
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import cPickle
import argparse


def data_shuffle(x, y):
	rng_state = rand.get_state()
	rand.shuffle(x)
	rand.set_state(rng_state)
	rand.shuffle(y)

parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
args = parser.parse_args()

print "loading data... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']
		
datafile.close()


# print "shuffling data... "
# data_shuffle(scores,positions)
shared_positions = theano.shared(positions.astype(theano.config.floatX), name="positions")
shared_scores = theano.shared(scores.astype(theano.config.floatX), name="scores")
n_train = shared_scores.get_value(borrow=True).shape[0]

indices = T.ivector(name="indices") #index of data
y = T.tensor3('y') #target output score

numEpochs = 100
iteration = 0
print_interval = 10
batch_size = 1
numBatches = n_train/batch_size

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	f.close()
else:
	print "building model..."
	network = network(batch_size=batch_size)

cost = T.mean(T.sqr(network.output - y))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
    [indices],
    cost,
    updates = updates,
    givens={
        network.input: shared_positions[indices],
        y: shared_scores[indices]
    }
)

test_model = theano.function(
    [indices],
    cost,
    givens={
        network.input: shared_positions[indices],
        y: shared_scores[indices]
    }
)

evaluate_model = theano.function(
	[indices],
	network.output,
	givens={
        network.input: shared_positions[indices],
    }
)


print "Training model on mentor set..."
indices = range(n_train)
for epoch in range(numEpochs):
	np.random.shuffle(indices)
	cost = 0
	for batch in range(numBatches):
		cost+=train_model(indices[batch*batch_size:(batch+1)*batch_size])
		iteration+=1
		print "Cost: ",cost/(batch+1)

print "done training!"

print "saving network..."
if args.save:
	f = file(args.save, 'wb')
else:
	f = file('mentor_network.save', 'wb')
cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()