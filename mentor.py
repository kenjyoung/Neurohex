import theano
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network


def data_shuffle(x, y):
	rng_state = rand.get_state()
	rand.shuffle(x)
	rand.set_state(rng_state)
	rand.shuffle(y)

print "loading data... "
datafile = open("data/scoredPositions.npz", 'r')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']
datafile.close()

# print "shuffling data... "
# data_shuffle(scores,positions)
shared_positions = theano.shared(positions.astype(theano.config.floatX), name="positions")
shared_scores = theano.shared(scores.astype(theano.config.floatX), name="scores")
# print shared_positions.shape.eval()
n_train = shared_scores.get_value(borrow=True).shape[0]

print "building model..."
index = T.iscalar(name="index") #index of data
y = T.tensor3('y') #target output score

numEpochs = 1000
iteration = 0
print_interval = 10
batch_size = 1
numBatches = n_train/batch_size

network = network(batch_size=batch_size)

cost = T.mean((network.output - y)**2)

#should tune parameters here at some point, this just uses defaults
updates = rmsprop(cost, network.params)

train_model = theano.function(
    [index],
    cost,
    updates = updates,
    givens={
        network.input: shared_positions[index*batch_size : (index+1)*batch_size],
        y: shared_scores[index*batch_size : (index+1)*batch_size]
    }
)

#TODO: fix shuffling
print "Training model on mentor set..."
for epoch in range(numEpochs):
	indices = range(numBatches)
	#np.random.shuffle(indices)
	cost = 0
	for i in indices:
		cost+=train_model(i)
		iteration+=1
		if iteration%print_interval == 0:
			print "Training Example: ", iteration
			print "Cost: ",cost/print_interval
			cost = 0

print "done training!"





