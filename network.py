import theano
from theano import tensor as T
import numpy as np
from inputFormat import *
from layers import *
import cPickle


class network:
	def __init__(self, batch_size = 1, rng = None, load_file = None):
		if(not rng): rng = np.random.RandomState(234567)
		self.input = T.tensor4('input') #position matrix
		layer0_D3 = 16
		layer0_D5 = 0

		layer0 = HexConvLayer(
			rng,
			self.input, 
			(batch_size, 6, input_size, input_size), 
			layer0_D5, 
			layer0_D3
		)

		layer1_D3 = 16
		layer1_D5 = 0

		layer1 = HexConvLayer(
			rng,
			layer0.output,
			(batch_size, layer0_D3+layer0_D5, input_size, input_size),
			layer1_D5,
			layer1_D3
		)

		layer2_D3 = 16
		layer2_D5 = 0

		layer2 = HexConvLayer(
			rng,
			layer1.output,
			(batch_size, layer1_D3+layer1_D5, input_size, input_size),
			layer2_D5,
			layer2_D3
			)

		layer3 = FullyConnectedLayer(
		 	rng,
		 	input = layer2.output.flatten(2),
		 	n_in = (layer2_D3+layer2_D5)*input_size*input_size,
		 	n_out = boardsize*boardsize
		)


		self.output = 2*T.nnet.sigmoid(layer3.output)-1

		self.output = self.output.reshape((batch_size, boardsize, boardsize))

		self.params = layer0.params + layer1.params + layer2.params +layer3.params






