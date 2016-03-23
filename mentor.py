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
import os

def save():
	print "saving network..."
	if args.save:
		save_name = args.save
	else:
		save_name = "mentor_network.save"
	if args.data:
		f = file(args.data+"/"+save_name, 'wb')
	else:
		f = file(save_name, 'wb')
	cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	if args.data:
		f = file(args.data+"/costs.save","wb")
		cPickle.dump(costs, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

print "loading data... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']

if args.data:
	if not os.path.exists(args.data):
		os.makedirs(args.data)
		costs = []
	else:
		if os.path.exists(args.data+"/costs.save"):
			f = file(args.data+"/costs.save")
			costs = cPickle.load(f)
			f.close
		else:
			costs = []
else:
	costs = []
		
datafile.close()

positions = positions.astype(theano.config.floatX)
scores = scores.astype(theano.config.floatX)
n_train = scores.shape[0]

positions_batch = T.tensor4('positions_batch')
y = T.tensor3('y') #target output score

numEpochs = 100
iteration = 0
batch_size = 64
numBatches = n_train/batch_size

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	if(network.batch_size):
		batch_size = network.batch_size
	f.close()
else:
	print "building model..."
	#use batchsize none now so that we can easily use same network for picking single moves and evaluating batches
	network = network(batch_size=None)

cost = T.mean(T.sqr(network.output.reshape((batch_size, boardsize, boardsize)) - y))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
    [positions_batch, y],
    cost,
    updates = updates,
    givens={
        network.input: positions_batch,
    }
)

test_model = theano.function(
    [positions_batch, y],
    cost,
    givens={
        network.input: positions_batch,
    }
)

evaluate_model = theano.function(
	[positions_batch],
	network.output,
	givens={
        network.input: positions_batch,
    }
)

costs = []

print "Training model on mentor set..."
indices = range(n_train)
try:
	for epoch in range(numEpochs):
		print "epoch: ",epoch
		np.random.shuffle(indices)
		cost_sum = 0
		for batch in range(numBatches):
			t = time.clock()
			p_batch = positions[indices[batch*batch_size:(batch+1)*batch_size]]
			s_batch = scores[indices[batch*batch_size:(batch+1)*batch_size]]
			cost=train_model(p_batch, s_batch)
			run_time = time.clock()-t
			cost_sum+=cost
			iteration+=1
			print "Cost: ",cost_sum/(batch+1), " Time per position: ", run_time/(batch_size)
		costs.append(cost_sum/(batch+1))
		plt.plot(costs)
		plt.ylabel('cost')
		plt.xlabel('epoch')
		plt.draw()
		plt.pause(0.001)
		#save snapshot of network every epoch in case something goes wrong
		save()
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save()
	exit(1)


print "done training!"

save()

cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()