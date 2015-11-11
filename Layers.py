import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np

class HexConvLayer:
    def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters):
        D3_W_bound = np.sqrt(6. / (7*(input_shape[1] + num_D3_filters)))
        D5_W_bound = np.sqrt(6. / (19*(input_shape[1] + num_D5_filters)))

        self.W3_values = theano.shared(np.asarray(rng.uniform(low=-D3_W_bound, high=D3_W_bound, size=(num_D3_filters,input_shape[1],7))), borrow = True)
        W3 = T.zeros((num_D3_filters,input_shape[1],3,3))
        T.set_subtensor(W3[:,:,1:,0], self.W3_values[:,:,:3])
        T.set_subtensor(W3[:,:,:,1], self.W3_values[:,:,3:6])
        T.set_subtensor(W3[:,:,:2,2], self.W3_values[:,:,6:])

        self.W5_values = theano.shared(np.asarray(rng.uniform(low=-D3_W_bound, high=D3_W_bound, size=(num_D3_filters,input_shape[1],19))), borrow = True)
        W5 = T.zeros((num_D5_filters,input_shape[1],5,5))
        T.set_subtensor(W5[:,:,2:,0], self.W5_values[:,:,:3])
        T.set_subtensor(W5[:,:,1:,1], self.W5_values[:,:,3:7])
        T.set_subtensor(W5[:,:,:,2], self.W5_values[:,:,7:12])
        T.set_subtensor(W5[:,:,:4,3], self.W5_values[:,:,12:15])
        T.set_subtensor(W5[:,:,:3,4], self.W5_values[:,:,15:])

        #TODO: possibly change to use position dependent biases
        b_values = np.zeros((num_D5_filters+num_D3_filters), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out3 = conv.conv2d(
            input = input[:,:,1:-1,1:-1],
            filters = W3,
            filter_shape = (num_D3_filters,input_shape[1],3,3),
            image_shape = input_shape
        )

        conv_out5 = conv.conv2d(
            input = input,
            filters = W5,
            filter_shape = (num_D5_filters,input_shape[1],5,5),
            image_shape = input_shape
        )

        full_out = T.concatenate([conv_out5, conv_out3], axis=1)

        squished_out = T.nnet.relu(full_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        padded_out = T.zeros((input_shape[0], num_D3_filters+num_D5_filters, input_shape[2], input_shape[3]))
        T.set_subtensor(padded_out[:,:,2:-2,2:-2], squished_out)

        self.output = padded_out

        self.params = [self.W5_values, self.W3_values, self.b]

        self.input = input

class FullyConnectedLayer:
    def __init__(self, rng, input, n_in, n_out):
        self.input = input
        W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W_values, name='W', borrow=True)


        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.output = T.nnet.relu(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

