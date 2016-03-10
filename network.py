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
		self.batch_size = batch_size
		layer0_D3 = 16
		layer0_D5 = 0

		layer0 = HexConvLayer(
			rng,
			self.input, 
			(batch_size, 6, input_size, input_size), 
			layer0_D5, 
			layer0_D3
		)

		layer1_D3 = 8
		layer1_D5 = 16

		layer1 = HexConvLayer(
			rng,
			layer0.output,
			(batch_size, layer0_D3+layer0_D5, input_size, input_size),
			layer1_D5,
			layer1_D3
		)

		layer2_D3 = 12
		layer2_D5 = 12

		layer2 = HexConvLayer(
			rng,
			layer1.output,
			(batch_size, layer1_D3+layer1_D5, input_size, input_size),
			layer2_D5,
			layer2_D3
		)

		layer3_D3 = 16
		layer3_D5 = 8

		layer3 = HexConvLayer(
			rng,
			layer2.output,
			(batch_size, layer2_D3+layer2_D5, input_size, input_size),
			layer3_D5,
			layer3_D3
		)

		layer4_D3 = 20
		layer4_D5 = 4

		layer4 = HexConvLayer(
			rng,
			layer3.output,
			(batch_size, layer3_D3+layer3_D5, input_size, input_size),
			layer4_D5,
			layer4_D3
		)

		layer5_D3 = 24
		layer5_D5 = 0

		layer5 = HexConvLayer(
			rng,
			layer4.output,
			(batch_size, layer4_D3+layer4_D5, input_size, input_size),
			layer5_D5,
			layer5_D3
		)


		layer3 = SigmoidLayer(
		 	rng,
		 	input = layer5.output.flatten(2),
		 	n_in = (layer5_D3+layer5_D5)*input_size*input_size,
		 	n_out = boardsize*boardsize
		)

		self.output = 2*layer3.output-1

		self.params = layer0.params + layer1.params + layer2.params +layer3.params






