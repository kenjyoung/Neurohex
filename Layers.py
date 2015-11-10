import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np

class HexConvLayer:
	def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters):
		D3_W_bound = numpy.sqrt(6. / (7*(input_shape[1] + num_D3_filters)))
		D5_W_bound = numpy.sqrt(6. / (19*(input_shape[1] + num_D5_filters)))

		W3_values = theano.shared(np.asarray(rng.uniform(low=-W3_bound, high=W3_bound, size=(num_D3_filters,7))), borrow = True)
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

		self.W5_values = theano.shared(np.asarray(rng.uniform(low=-W3_bound, high=W3_bound, size=(num_D3_filters,19))), borrow = True)
		W5_array = np.zeros((num_D5_filters,5,5))

		for i in range(num_D5_filters):
			W5_array[i,2:,0] = W5_values[i,:3]
			W5_array[i,1:,1] = W5_values[i,3:7]
			W5_array[i,:,2]  = W5_values[i,7:12]
			W5_array[i,:4,3] = W5_values[i,12:15]
			W5_array[i,:3,4] = W5_values[i,15:]

		W5 = theano.shared(
			W5_array,
			borrow=True
		)

		#TODO: possibly change to use position dependent biases
		b_values = numpy.zeros((num_D5_filters+num_D3_filters), dtyper=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		conv_out3 = conv.conv2d(
			input = input[:,:,1:-1,1:-1],
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

		squished_out = T.relu(full_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		padded_out = T.zeros(inputshape[2:])
		padded_out.set_subtensor(padded_out[:,:,2:-2,2:-2], squished_out)

		self.output = padded_out

		self.params = [self.W5_values, self.W3_values, self.b]

		self.input = input

class FullyConnectedLayer:
	def __init__(self, rng, input, n_in, n_out):
		self.input = input
		W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

		self.W = theano.shared(value=W_values, name='W', borrow=True)
		b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.output = T.relu(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

