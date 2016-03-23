import cPickle
import argparse
from inputFormat import *
from network import network


parser = argparse.ArgumentParser()
parser.add_argument("source", type=str, help="Pickled network to steal params from.")
parser.add_argument("dest", type=str, help="File to place new network in.")
args = parser.parse_args()

print "loading model..."
f = file(args.source, 'rb')
old_network = cPickle.load(f)
f.close()

new_network = network(batch_size=None, params = old_network.params)

print "saving model..."
f = file(args.dest, 'wb')
cPickle.dump(new_network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()