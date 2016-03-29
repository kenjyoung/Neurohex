import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np

class HexConvLayer:
    def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters, params = None):
        W3_bound = np.sqrt(6. / (7*(input_shape[1] + num_D3_filters)))
        W5_bound = np.sqrt(6. / (19*(input_shape[1] + num_D5_filters)))

        if(params):
        	self.W3_values = params[1]
        else:
	        self.W3_values = theano.shared(
	            np.asarray(
	                rng.uniform(
	                    low=-W3_bound,
	                    high=W3_bound,
	                    size=(num_D3_filters,input_shape[1],7)
	                ),
	                dtype=theano.config.floatX
	            ),
	            borrow = True
	        )
        #Place weights in hexagonal filter of diameter 3
        W3 = T.zeros((num_D3_filters,input_shape[1],3,3))
        W3 = T.set_subtensor(W3[:,:,1:,0], self.W3_values[:,:,:2])
        W3 = T.set_subtensor(W3[:,:,:,1], self.W3_values[:,:,2:5])
        W3 = T.set_subtensor(W3[:,:,:2,2], self.W3_values[:,:,5:])

        if(params):
        	self.W5_values = params[0]
        else:
	        self.W5_values = theano.shared(
	            np.asarray(
	                rng.uniform(
	                    low=-W5_bound,
	                    high=W5_bound,
	                    size=(num_D5_filters,input_shape[1],19)
	                ),
	                dtype=theano.config.floatX
	            ),
	            borrow = True
	        )
        #Place weights in hexagonal filter of diameter 5
        W5 = T.zeros((num_D5_filters,input_shape[1],5,5))
        W5 = T.set_subtensor(W5[:,:,2:,0], self.W5_values[:,:,:3])
        W5 = T.set_subtensor(W5[:,:,1:,1], self.W5_values[:,:,3:7])
        W5 = T.set_subtensor(W5[:,:,:,2], self.W5_values[:,:,7:12])
        W5 = T.set_subtensor(W5[:,:,:4,3], self.W5_values[:,:,12:16])
        W5 = T.set_subtensor(W5[:,:,:3,4], self.W5_values[:,:,16:])

        if(params):
        	self.b = params[2]
        else:
	        b_values = np.zeros((num_D5_filters+num_D3_filters), dtype=theano.config.floatX)
	    	self.b = theano.shared(value=b_values, borrow=True)

        conv_out3 = conv.conv2d(
            input = input[:,:,1:-1,1:-1],
            filters = W3,
            filter_shape = (num_D3_filters,input_shape[1],3,3),
            image_shape = [input_shape[0], input_shape[1], input_shape[2]-2, input_shape[3]-2]
        )

        conv_out5 = conv.conv2d(
            input = input,
            filters = W5,
            filter_shape = (num_D5_filters,input_shape[1],5,5),
            image_shape = input_shape
        )

        full_out = T.concatenate([conv_out5, conv_out3], axis=1)

        squished_out = T.nnet.relu(full_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        padded_out = T.zeros((squished_out.shape[0], num_D3_filters+num_D5_filters, input_shape[2], input_shape[3]))
        padded_out = T.set_subtensor(padded_out[:,:,2:-2,2:-2], squished_out)

        self.output = padded_out

        self.params = [self.W5_values, self.W3_values, self.b]

        self.mem_size = (T.prod(self.W5_values.shape)+T.prod(self.W3_values.shape)+T.prod(self.b.shape))*4

        self.input = input

class HexConvLayerBN:
    def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters, params = None, alpha = 0.25):
        W3_bound = np.sqrt(6. / (7*(input_shape[1] + num_D3_filters)))
        W5_bound = np.sqrt(6. / (19*(input_shape[1] + num_D5_filters)))
        self.alpha = alpha

        if(params):
        	self.W3_values = params[1]
        else:
	        self.W3_values = theano.shared(
	            np.asarray(
	                rng.uniform(
	                    low=-W3_bound,
	                    high=W3_bound,
	                    size=(num_D3_filters,input_shape[1],7)
	                ),
	                dtype=theano.config.floatX
	            ),
	            borrow = True
	        )
        #Place weights in hexagonal filter of diameter 3
        W3 = T.zeros((num_D3_filters,input_shape[1],3,3))
        W3 = T.set_subtensor(W3[:,:,1:,0], self.W3_values[:,:,:2])
        W3 = T.set_subtensor(W3[:,:,:,1], self.W3_values[:,:,2:5])
        W3 = T.set_subtensor(W3[:,:,:2,2], self.W3_values[:,:,5:])

        if(params):
        	self.W5_values = params[0]
        else:
	        self.W5_values = theano.shared(
	            np.asarray(
	                rng.uniform(
	                    low=-W5_bound,
	                    high=W5_bound,
	                    size=(num_D5_filters,input_shape[1],19)
	                ),
	                dtype=theano.config.floatX
	            ),
	            borrow = True
	        )
        #Place weights in hexagonal filter of diameter 5
        W5 = T.zeros((num_D5_filters,input_shape[1],5,5))
        W5 = T.set_subtensor(W5[:,:,2:,0], self.W5_values[:,:,:3])
        W5 = T.set_subtensor(W5[:,:,1:,1], self.W5_values[:,:,3:7])
        W5 = T.set_subtensor(W5[:,:,:,2], self.W5_values[:,:,7:12])
        W5 = T.set_subtensor(W5[:,:,:4,3], self.W5_values[:,:,12:16])
        W5 = T.set_subtensor(W5[:,:,:3,4], self.W5_values[:,:,16:])

        if(params):
        	self.beta = params[2]
        	self.gamma = params[3]
        else:
        	self.gamma = theano.shared(np.zeros((num_D5_filters+num_D3_filters), dtype=theano.config.floatX), borrow=True)
	        self.beta = theano.shared(np.zeros((num_D5_filters+num_D3_filters), dtype=theano.config.floatX), borrow=True)

        conv_out3 = conv.conv2d(
            input = input[:,:,1:-1,1:-1],
            filters = W3,
            filter_shape = (num_D3_filters,input_shape[1],3,3),
            image_shape = [input_shape[0], input_shape[1], input_shape[2]-2, input_shape[3]-2]
        )

        conv_out5 = conv.conv2d(
            input = input,
            filters = W5,
            filter_shape = (num_D5_filters,input_shape[1],5,5),
            image_shape = input_shape
        )

        self.full_out = T.concatenate([conv_out5, conv_out3], axis=1)
        mean = self.full_out.mean(axis = (0,2,3))
		std = self.full_out.std(axis = (0,2,3))

        self.alpha = alpha
    	running_mean = theano.clone(mean, share_inputs=False)
        running_std = theano.clone(std, share_inputs=False)
    	self.running_mean.default_update = ((1-self.alpha)*self.running_mean+self.alpha*mean)
    	self.running_std.default_update = ((1-self.alpha)*self.running_std+self.alpha*std)

        self.params = [self.W5_values, self.W3_values, self.beta, self.gamma]

        self.mem_size = (T.prod(self.W5_values.shape)+T.prod(self.W3_values.shape)+T.prod(self.beta.shape)+T.prod(self.gamma.shape))*4

        self.input = input

        def get_output(self, deterministic = False):
	    	#test time output
	    	if(deterministic):
	        	normed_out= batch_normalization(self.full_out, 
							self.gamma.dimshuffle('x',0,'x','x'),
							self.beta.dimshuffle('x',0,'x','x'), 
							self.running_mean.dimshuffle('x',0,'x','x'), 
							self.running_std.dimshuffle('x',0,'x','x'))
	        #training time output
			else:
				mean = self.full_out.mean(axis = (0,2,3))
				std = self.full_out.std(axis = (0,2,3))
				# include these in computation graph so that their default update will be applied by any function that uses mean and std
			    # (this idea borrowed from lasagne)
			    mean += 0 * self.running_mean
			    std += 0 * self.running_std
				normed_out= batch_normalization(self.full_out, 
							self.gamma.dimshuffle('x',0,'x','x'),
							self.beta.dimshuffle('x',0,'x','x'), 
							mean.dimshuffle('x',0,'x','x'), 
							std.dimshuffle('x',0,'x','x'))
			squished_out = T.nnet.relu(normed_out)

			padded_out = T.zeros((input_shape[0], num_D3_filters+num_D5_filters, input_shape[2], input_shape[3]))
			padded_out = T.set_subtensor(padded_out[:,:,2:-2,2:-2], squished_out)
			return padded_out


class FullyConnectedLayer:
    def __init__(self, rng, input, n_in, n_out, params = None):
		self.input = input
		if(params):
			self.W = params[0]
		else:
		    W_values = np.asarray(
		            rng.uniform(
		                low=-np.sqrt(6. / (n_in + n_out)),
		                high=np.sqrt(6. / (n_in + n_out)),
		                size=(n_in, n_out)
		            ),
		            dtype=theano.config.floatX
		        )
		    self.W = theano.shared(value=W_values, name='W', borrow=True)

		if(params):
			self.b = params[1]
		else:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, name='b', borrow=True)

		self.output = T.nnet.relu(T.dot(input, self.W) + self.b)

		self.params = [self.W, self.b]

		self.mem_size = (T.prod(self.W.shape)+T.prod(self.b.shape))*4

class SigmoidLayer:
    def __init__(self, rng, input, n_in, n_out, params = None):
		self.input = input
		if(params):
			self.W = params[0]
		else:
		    W_values = np.asarray(
		            rng.uniform(
		                low=-np.sqrt(6. / (n_in + n_out)),
		                high=np.sqrt(6. / (n_in + n_out)),
		                size=(n_in, n_out)
		            ),
		            dtype=theano.config.floatX
		        )
		    self.W = theano.shared(value=W_values, name='W', borrow=True)

		if(params):
			self.b = params[1]
		else:
		    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		    self.b = theano.shared(value=b_values, name='b', borrow=True)

		self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

		self.params = [self.W, self.b]

		self.mem_size = (T.prod(self.W.shape)+T.prod(self.b.shape))*4

class SigmoidLayerBN:
    def __init__(self, rng, input, n_in, n_out, params = None, alpha = 0.25):
		self.input = input
		if(params):
			self.W = params[0]
		else:
		    W_values = np.asarray(
		            rng.uniform(
		                low=-np.sqrt(6. / (n_in + n_out)),
		                high=np.sqrt(6. / (n_in + n_out)),
		                size=(n_in, n_out)
		            ),
		            dtype=theano.config.floatX
		        )
		    self.W = theano.shared(value=W_values, name='W', borrow=True)

		if(params):
			self.beta = params[1]
			self.gamma = params[2]
		else:
		    self.beta = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)
		    self.gamma = theano.shared(np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)

		self.output = T.dot(self.input, self.W)

		self.params = [self.W, self.beta, self.gamma]

		self.mem_size = (T.prod(self.W.shape)+T.prod(self.b.shape))*4

	def get_output(self, deterministic = False):
	    	#test time output
	    	if(deterministic):
	        	normed_out= batch_normalization(self.output, 
							self.gamma,
							self.beta, 
							self.running_mean, 
							self.running_std)
	        #training time output
			else:
				mean = self.full_out.mean()
				std = self.full_out.std()
				# include these in computation graph so that their default update will be applied by any function that uses mean and std
			    # (this idea borrowed from lasagne)
			    mean += 0 * self.running_mean
			    std += 0 * self.running_std
				normed_out= batch_normalization(self.output, 
							self.gamma,
							self.beta, 
							mean, 
							std)
			return T.nnet.sigmoid(normed_out)


