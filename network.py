import theano
from theano import tensor as T
import numpy as np
from inputFormat import *
from layers import *
import cPickle


class network:
	def __init__(self, batch_size = None, rng = None, load_file = None, params = None):
		if(not rng): rng = np.random.RandomState(None)
		self.input = T.tensor4('input') #position matrix
		self.batch_size = batch_size
		layer0_D3 = 48
		layer0_D5 = 80

		layer0 = HexConvLayer(
			rng,
			self.input, 
			(batch_size, num_channels, input_size, input_size), 
			layer0_D5, 
			layer0_D3,
			params = params[0:3] if params else None
		)

		layer1_D3 = 64
		layer1_D5 = 64

		layer1 = HexConvLayer(
			rng,
			layer0.output,
			(batch_size, layer0_D3+layer0_D5, input_size, input_size),
			layer1_D5,
			layer1_D3,
			params[3:6] if params else None
		)

		layer2_D3 = 80
		layer2_D5 = 48

		layer2 = HexConvLayer(
			rng,
			layer1.output,
			(batch_size, layer1_D3+layer1_D5, input_size, input_size),
			layer2_D5,
			layer2_D3,
			params[6:9] if params else None
		)

		layer3_D3 = 96
		layer3_D5 = 32

		layer3 = HexConvLayer(
			rng,
			layer2.output,
			(batch_size, layer2_D3+layer2_D5, input_size, input_size),
			layer3_D5,
			layer3_D3,
			params[9:12] if params else None
		)

		layer4_D3 = 112
		layer4_D5 = 16

		layer4 = HexConvLayer(
			rng,
			layer3.output,
			(batch_size, layer3_D3+layer3_D5, input_size, input_size),
			layer4_D5,
			layer4_D3,
			params[12:15] if params else None
		)

		layer5_D3 = 128
		layer5_D5 = 0

		layer5 = HexConvLayer(
			rng,
			layer4.output,
			(batch_size, layer4_D3+layer4_D5, input_size, input_size),
			layer5_D5,
			layer5_D3,
			params[15:18] if params else None
		)

		layer6_D3 = 128
		layer6_D5 = 0

		layer6 = HexConvLayer(
			rng,
			layer5.output,
			(batch_size, layer5_D3+layer5_D5, input_size, input_size),
			layer6_D5,
			layer6_D3,
			params[18:21] if params else None
		)

		layer7_D3 = 128
		layer7_D5 = 0

		layer7 = HexConvLayer(
			rng,
			layer6.output,
			(batch_size, layer6_D3+layer6_D5, input_size, input_size),
			layer7_D5,
			layer7_D3,
			params[21:24] if params else None
		)

		layer8_D3 = 128
		layer8_D5 = 0

		layer8 = HexConvLayer(
			rng,
			layer7.output,
			(batch_size, layer7_D3+layer7_D5, input_size, input_size),
			layer8_D5,
			layer8_D3,
			params[24:27] if params else None
		)

		layer9_D3 = 128
		layer9_D5 = 0

		layer9 = HexConvLayer(
			rng,
			layer8.output,
			(batch_size, layer8_D3+layer8_D5, input_size, input_size),
			layer9_D5,
			layer9_D3,
			params[27:30] if params else None
		)

		layer10 = SigmoidLayer(
		 	rng,
		 	layer9.output.flatten(2),
		 	(layer9_D3+layer9_D5)*input_size*input_size,
		 	boardsize*boardsize,
		 	params[30:32] if params else None
		)

		self.output = 2*layer10.output-1

		self.params = layer0.params + layer1.params + layer2.params +layer3.params + layer4.params + layer5.params + layer6.params+ layer7.params + layer8.params + layer9.params + layer10.params

		self.mem_size = layer1.mem_size+layer2.mem_size+layer3.mem_size+layer4.mem_size+layer5.mem_size+layer6.mem_size+ layer7.mem_size + layer8.mem_size + layer9.mem_size + layer10.mem_size


class policy_network:
	def __init__(self, batch_size = None, rng = None, load_file = None, params = None):
		if(not rng): rng = np.random.RandomState(None)
		self.input = T.tensor4('input') #position matrix
		self.batch_size = batch_size
		layer0_D3 = 48
		layer0_D5 = 80

		layer0 = HexConvLayer(
			rng,
			self.input, 
			(batch_size, num_channels, input_size, input_size), 
			layer0_D5, 
			layer0_D3,
			params = params[0:3] if params else None
		)

		layer1_D3 = 64
		layer1_D5 = 64

		layer1 = HexConvLayer(
			rng,
			layer0.output,
			(batch_size, layer0_D3+layer0_D5, input_size, input_size),
			layer1_D5,
			layer1_D3,
			params[3:6] if params else None
		)

		layer2_D3 = 80
		layer2_D5 = 48

		layer2 = HexConvLayer(
			rng,
			layer1.output,
			(batch_size, layer1_D3+layer1_D5, input_size, input_size),
			layer2_D5,
			layer2_D3,
			params[6:9] if params else None
		)

		layer3_D3 = 96
		layer3_D5 = 32

		layer3 = HexConvLayer(
			rng,
			layer2.output,
			(batch_size, layer2_D3+layer2_D5, input_size, input_size),
			layer3_D5,
			layer3_D3,
			params[9:12] if params else None
		)

		layer4_D3 = 112
		layer4_D5 = 16

		layer4 = HexConvLayer(
			rng,
			layer3.output,
			(batch_size, layer3_D3+layer3_D5, input_size, input_size),
			layer4_D5,
			layer4_D3,
			params[12:15] if params else None
		)

		layer5_D3 = 128
		layer5_D5 = 0

		layer5 = HexConvLayer(
			rng,
			layer4.output,
			(batch_size, layer4_D3+layer4_D5, input_size, input_size),
			layer5_D5,
			layer5_D3,
			params[15:18] if params else None
		)

		layer6_D3 = 128
		layer6_D5 = 0

		layer6 = HexConvLayer(
			rng,
			layer5.output,
			(batch_size, layer5_D3+layer5_D5, input_size, input_size),
			layer6_D5,
			layer6_D3,
			params[18:21] if params else None
		)

		layer7_D3 = 128
		layer7_D5 = 0

		layer7 = HexConvLayer(
			rng,
			layer6.output,
			(batch_size, layer6_D3+layer6_D5, input_size, input_size),
			layer7_D5,
			layer7_D3,
			params[21:24] if params else None
		)

		layer8_D3 = 128
		layer8_D5 = 0

		layer8 = HexConvLayer(
			rng,
			layer7.output,
			(batch_size, layer7_D3+layer7_D5, input_size, input_size),
			layer8_D5,
			layer8_D3,
			params[24:27] if params else None
		)

		layer9_D3 = 128
		layer9_D5 = 0

		layer9 = HexConvLayer(
			rng,
			layer8.output,
			(batch_size, layer8_D3+layer8_D5, input_size, input_size),
			layer9_D5,
			layer9_D3,
			params[27:30] if params else None
		)

		layer10 = FullyConnectedLayer(
		 	rng,
		 	layer9.output.flatten(2),
		 	(layer9_D3+layer9_D5)*input_size*input_size,
		 	boardsize*boardsize,
		 	params[30:32] if params else None
		)

		not_played = T.and_(T.eq(self.input[:, white, padding:boardsize+padding, padding:boardsize+padding].flatten(2),0),\
		      	T.eq(self.input[:, black, padding:boardsize+padding, padding:boardsize+padding].flatten(2),0))

		playable_output = T.nnet.softmax(layer10.output[not_played.nonzero()])

		output = T.switch(not_played, layer10.output, -1*np.inf)

		self.output = T.nnet.softmax(output)

		self.params = layer0.params + layer1.params + layer2.params +layer3.params + layer4.params + layer5.params + layer6.params+ layer7.params + layer8.params + layer9.params + layer10.params

		self.mem_size = layer1.mem_size+layer2.mem_size+layer3.mem_size+layer4.mem_size+layer5.mem_size+layer6.mem_size+ layer7.mem_size + layer8.mem_size + layer9.mem_size + layer10.mem_size







