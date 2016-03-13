import theano
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import cPickle
import argparse

def policy(state, evaluator):
	rand = np.random.random()
	played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
	if(rand>epsilon_q):
		scores = evaluator(state)
		#set value of played cells impossibly low so they are never picked
		scores[played] = -2
		return scores.argmax()
	#choose random open cell
	return np.random.choice(np.arange(boardsize*boardsize)[np.logical_not(played)])

def Q_update():
	batch = np.random.choice(np.arange(0,replay_size), size=(batch_size))
	states1 = state1_memory[batch]
	states2 = state2_memory[batch]
	scores = evaluate_model_batch(states2)
	played = np.logical_or(states2[:,white,padding:boardsize+padding,padding:boardsize+padding],\
		     states2[:,black,padding:boardsize+padding,padding:boardsize+padding]).reshape(scores.shape)
	#set value of played cells impossibly low so they are never picked
	scores[played] = -2
	actions = action_memory[batch]
	rewards = reward_memory[batch]
	targets = np.zeros(rewards.size).astype(theano.config.floatX)
	targets[rewards==1] = 1
	targets[rewards==0] = -np.amax(scores, axis=1)[rewards==0]
	cost = train_model(states1,targets,actions)
	return cost

def action_to_cell(action):
	cell = np.unravel_index(action, (boardsize,boardsize))
	return(cell[0]+padding, cell[1]+padding)
	
parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
args = parser.parse_args()

print "loading starting positions... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

input_state = T.tensor3('input_state')

state_batch = T.tensor4('state_batch')
target_batch = T.dvector('target_batch')
action_batch = T.ivector('action_batch')


replay_capacity = 10000
replay_size = 0

#replay memory from which updates are drawn
replay_index = 0
replay_full = False
state1_memory = np.zeros(np.concatenate(([replay_capacity], input_shape)), dtype=bool)
action_memory = np.zeros(replay_capacity, dtype=np.int8)
reward_memory = np.zeros(replay_capacity, dtype=bool)
state2_memory = np.zeros(np.concatenate(([replay_capacity], input_shape)), dtype=bool)

numEpisodes = 100000
batch_size = 32

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	batch_size = network.batch_size
	f.close()
else:
	print "building model..."
	network = network(batch_size=batch_size)

#zeros used for running network on a single state without modifying batch size
input_padding = theano.shared(np.zeros(np.concatenate(([network.batch_size],input_shape))).astype(theano.config.floatX))
evaluate_model_single = theano.function(
	[input_state],
	network.output[0],
	givens={
        network.input: T.set_subtensor(input_padding[0,:,:,:], input_state),
	}
)

evaluate_model_batch = theano.function(
	[state_batch],
	network.output,
	givens={
        network.input: state_batch,
	}
)

cost = T.mean(T.sqr(network.output[T.arange(target_batch.shape[0]),action_batch] - target_batch))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
	[state_batch,target_batch,action_batch],
	cost,
	updates = updates,
	givens={
		network.input: state_batch,
	}
)

print "Running episodes..."
epsilon_q=0.1
try:
	for i in range(numEpisodes):
		cost = 0
		num_step = 1
		#randomly choose who is to move from each position to increase variability in dataset
		move_parity = np.random.choice([True,False])
		#randomly choose starting position from database
		index = np.random.randint(numPositions)
		gameW = np.copy(positions[index])
		gameB = mirror_game(gameW)
		while(winner(gameW)==None):
			action = policy(gameW if move_parity else gameB, evaluate_model_single)
			action_memory[replay_index] = action
			state1_memory[replay_index,:,:] = np.copy(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
			play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
			play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
			if(not winner(gameW)==None):
				#only the player who just moved can win, so if anyone wins the reward is 1
				#for the current player
				reward_memory[replay_index] = 1
			else:
				reward_memory[replay_index] = 0

			state2_memory[replay_index,:,:] = np.copy(gameB if move_parity else gameW)
			move_parity = not move_parity
			replay_index+=1
			if(replay_index>=replay_capacity):
				replay_full = True
				replay_index = 0
			if(replay_full):
				cost += Q_update()
				print state_string(gameW)
			elif(replay_size > batch_size):
				replay_size+=1
				cost += Q_update()
				print state_string(gameW)
			else:
				replay_size+=1
		print "Episode", i, "complete, cost: ", cost/num_step
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	print "saving network..."
	if args.save:
		f = file(args.save, 'wb')
	else:
		f = file('Q_training.save', 'wb')
	cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()





print "saving network..."
if args.save:
	f = file(args.save, 'wb')
else:
	f = file('Q_network.save', 'wb')
cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()