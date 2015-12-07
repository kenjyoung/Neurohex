import theano
from theano import tensor as T
import numpy as np
import numpy.random as rand
from resistance import score
from preprocess import *
from inputFormat import *
from Layers import *


def data_shuffle(x, y):
	rng_state = rand.get_state()
	rand.shuffle(x)
	rand.set_state(rng_state)
	rand.shuffle(y)

rng = np.random.RandomState(23455)
alpha = 0.01

print "loading data... "
datafile = open("data/scoredPositions.npz", 'r')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']
datafile.close()

print "shuffling data... "
data_shuffle(scores,positions)
shared_positions = theano.shared(positions.astype(theano.config.floatX), name="positions")
shared_scores = theano.shared(scores.astype(theano.config.floatX), name="scores")
print shared_positions.shape.eval()
n_train = shared_scores.get_value(borrow=True).shape[0]

print "building model..."
index = T.iscalar(name="index") #index of data
x = T.tensor3('x') #position matrix
y = T.matrix('y') #target output score

layer0_input = x.reshape((1, 6, input_size, input_size))
layer0_D3 = 8
layer0_D5 = 24

layer0 = HexConvLayer(
	rng,
	layer0_input, 
	(1, 6, input_size, input_size), 
	layer0_D5, 
	layer0_D3
)

layer1_D3 = 16
layer1_D5 = 16

layer1 = HexConvLayer(
	rng,
	layer0.output,
	(1, layer0_D3+layer0_D5, input_size, input_size),
	layer1_D5,
	layer1_D3
)

layer2 = FullyConnectedLayer(
 	rng,
 	input = layer1.output.flatten(),
 	n_in = (layer1_D3+layer1_D5)*input_size*input_size,
 	n_out = boardsize*boardsize
)

output = T.nnet.sigmoid(layer2.output)

params = layer0.params + layer1.params + layer2.params

cost = T.mean((output - y.flatten())**2)

grads = T.grad(cost, params)

updates = [
    (param_i, param_i-alpha * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates = updates,
    givens={
        x: shared_positions[index],
        y: shared_scores[index]
    }
)

numEpochs = 1

#ToDo:shuffle data after every epoch
for epoch in range(numEpochs):
	for i in range(n_train):
		print "example "+str(i)
		train_model(i)

print "done training!"





