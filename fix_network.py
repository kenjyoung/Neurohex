import cPickle
import argparse
from inputFormat import *
from network import network
from theano import tensor as T


parser = argparse.ArgumentParser()
parser.add_argument("source", type=str, help="Pickled network to steal params from.")
parser.add_argument("dest", type=str, help="File to place new network in.")
parser.add_argument("--cpu", "-c", dest="cpu", action='store_const',
					const=True, default=False,
					help="Convert network to run on a CPU.")
args = parser.parse_args()

print "loading model..."
f = file(args.source, 'rb')
old_network = cPickle.load(f)
f.close()

params = old_network.params
if args.cpu:
	print "converting gpu parameters..."
	new_params=[]
	for param in params:
		param = T._shared(param.get_value())
		new_params.append(param)
	params = new_params

new_network = network(batch_size=None, params = params)

print "saving model..."
f = file(args.dest, 'wb')
cPickle.dump(new_network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()