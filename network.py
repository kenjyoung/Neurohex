import theano
from theano import tensor as T
import numpy as np
import numpy.random as rand
from resistance import score
from preprocess import *
from inputFormat import *
from Layers import *


rng = numpy.random.RandomState(23455)

print "preprocessing data... "
positions = preprocess("data/raw_games_small.dat")
resistance_scores = []

print "computing resistance scores..."
for position in positions:
	resistance_scores.append(score(position, white))

print "shuffling data... "
scored_positions = zip(positions, resistance_scores)
rand.shuffle(scored_positions)

x = T.matrix('x') #position matrix
y = T.matrix('y') #target output score

layer0_input = x.reshape(1, 6, input_size, input_size)
layer0_D3 = 16
layer0_D5 = 48

layer0 = HexConvLayer(
	rng,
	layer0_input, 
	(1, 6, input_size, input_size), 
	layer0_D5, 
	layer0_D3
)

layer1_D3 = 32
layer1_D5 = 32

layer1 = HexConvLayer(
	rng,
	layer0.output,
	(1, layer0_D3+layer0_D5, input_size, input_size)
	layer1_D5,
	layer1_D3
)

layer2 = FullyConnectedLayer(
	rng,
	input = layer1.output.flatten(),
	n_in = (layer1_D3+layer1_D5)*input_size*input_size
	n_out = boardsize*boardsize
)

output = sigmoid(layer2.output)

params = layer0.params + layer1.params + layer2.params

cost = T.mean(T.sqr(output - y.flatten()))

grads = T.grad(cost, params)





