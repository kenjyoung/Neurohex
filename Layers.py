import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np

class HexConvLayer:
	def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters, num_input_maps):
		D3_W_bound = numpy.sqrt(6. / (7*(num_input_maps + num_D3_filters)))
		D5_W_bound = numpy.sqrt(6. / (19*(num_input_maps + num_D5_filters)))

		W3_values = rng.uniform(low=-W3_bound, high=W3_bound, size=(num_D3_filters,7))
		W3_array = np.zeros((num_D3_filters,3,3))

		#TODO: improve on the way this convertion is handled
		for i in range(num_D3_filters):
			W3_array[i,1:,0] = W3_values[i,:3]
			W3_array[i,:,1]  = W3_values[i,3:6]
			W3_array[i,:2,2] = W3_values[i,6:]


		self.W3 = theano.shared(
			W3_array,
			borrow=True
		)

		W5_values = rng.uniform(low=-W3_bound, high=W3_bound, size=(num_D3_filters,19))
		W5_array = np.zeros((num_D5_filters,5,5))

		for i in range(num_D5_filters):
			W5_array[i,2:,0] = W5_values[i,:3]
			W5_array[i,1:,1] = W5_values[i,3:7]
			W5_array[i,:,2]  = W5_values[i,7:12]
			W5_array[i,:4,3] = W5_values[i,12:15]
			W5_array[i,:3,4] = W5_values[i,15:]

		self.W5 = theano.shared(
			W5_array,
			borrow=True
		)

		#TODO: possibly change to use position dependent biases
		b_values = numpy.zeros((num_D5_filters+num_D3_filters), dtyper=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		conv_out3 = conv.conv2d(
			input = input,
			filters = self.W3,
			filter_shape = (num_D3_filters,3,3),
			input_shape = input_shape
		)

		conv_out5 = conv.conv2d(
			input = input,
			filters = self.W5,
			filter_shape = (num_D5_filters,5,5),
			image_shape = input_shape
		)

		full_out = T.concatenate([conv_out5, conv_out3], axis=1)

		self.output = T.relu(full_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.params = [self.W5, self.W3, self.b]

		self.input = input