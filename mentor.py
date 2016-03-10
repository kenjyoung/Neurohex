import theano
import time
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import matplotlib.pyplot as plt
import cPickle
import argparse


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
shared_positions = theano.tensor._shared(positions.astype(theano.config.floatX), name="positions", borrow=True)
shared_scores = theano.tensor._shared(scores.astype(theano.config.floatX), name="scores", borrow=True)
n_train = shared_scores.get_value(borrow=True).shape[0]

indices = T.ivector("indices") #index of data
y = T.tensor3('y') #target output score

numEpochs = 100
iteration = 0
batch_size = 32
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

cost = T.mean(T.sqr(network.output.reshape((batch_size, boardsize, boardsize)) - y))

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

epoch_cost = []

print "Training model on mentor set..."
indices = range(n_train)
try:
	for epoch in range(numEpochs):
		print "epoch: ",epoch
		np.random.shuffle(indices)
		cost_sum = 0
		for batch in range(numBatches):
			t = time.clock()
			cost=train_model(indices[batch*batch_size:(batch+1)*batch_size])
			run_time = time.clock()-t
			cost_sum+=cost
			iteration+=1
			print "Cost: ",cost_sum/(batch+1), " Time per position: ", run_time/(batch_size)
		epoch_cost.append(cost_sum/(batch+1))
		plt.plot(epoch_cost)
		plt.ylabel('cost')
		plt.xlabel('epoch')
		plt.draw()
		plt.pause(0.001)
		#save snapshot of network every epoch in case something goes wrong
		print "saving network..."
		if args.save:
			f = file(args.save, 'wb')
		else:
			f = file('mentor_training.save', 'wb')
		cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		#save learning curve
		f = file('learning_curve.dat', 'w')
		for item in epoch_cost:
			f.write("%f\n" % item)
		f.close()
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	print "saving network..."
	if args.save:
		f = file(args.save, 'wb')
	else:
		f = file('mentor_training.save', 'wb')
	cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


print "done training!"

print "saving network..."
if args.save:
	f = file(args.save, 'wb')
else:
	f = file('mentor_network.save', 'wb')

cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()