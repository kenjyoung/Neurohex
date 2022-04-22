import argparse
from program import Program
import threading
from gamestate import gamestate
import sys
from allInputsAgent import allInputsAgent
from twoInputsAgent import twoInputsAgent
from sixInputsAgent import sixInputsAgent
from fourteenInputsAgent import fourteenInputsAgent


class agent:
	def __init__(self, exe):
		self.exe = exe 
		self.program = Program(self.exe, True)
		self.name = self.program.sendCommand("name").strip()
		self.lock  = threading.Lock()

	def sendCommand(self, command):
		self.lock.acquire()
		answer = self.program.sendCommand(command)
		self.lock.release()
		return answer

	def reconnect(self):
		self.program.terminate()
		self.program = Program(self.exe,True)
		self.lock = threading.Lock()



def move_to_cell(move):
	x =	ord(move[0].lower())-ord('a')
	y = int(move[1:])-1
	return (x,y)

def cell_to_move(cell):
	return chr(ord('a')+cell[0]-2)+str(cell[1]-2+1)


def run_game(blackAgent, whiteAgent, boardsize, verbose = False, opening = None):
	game = gamestate(boardsize)
	winner = None
	moves = []
	game.place_black(move_to_cell(opening))
	blackAgent.set_gamestate(game)
	whiteAgent.set_gamestate(game)
	whiteAgent.resetInput()
	blackAgent.resetInput()
	whiteAgent.openingMove(move_to_cell(opening),1)
	blackAgent.openingMove(move_to_cell(opening),1)
	move = whiteAgent.best_move()
    	
	moves.append(cell_to_move(move))
	game.place_white(move)
	blackAgent.move(move,0)
	whiteAgent.move(move,0)
	if verbose:
		print((blackAgent.name+" v.s. "+whiteAgent.name))
		print(game)
	if(game.winner() != game.PLAYERS["none"]):
		winner = game.winner()
		return winner
	
	while(True):
		move = blackAgent.best_move()
		moves.append(cell_to_move(move))
		game.place_black(move)
		whiteAgent.move(move,1)
		blackAgent.move(move,1)
		if verbose:
			print((blackAgent.name+" v.s. "+whiteAgent.name))
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
		move = whiteAgent.best_move()
		moves.append(cell_to_move(move))
		game.place_white(move)
		blackAgent.move(move,0)
		whiteAgent.move(move,0)
		if verbose:
			print((blackAgent.name+" v.s. "+whiteAgent.name))
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
	winner_name = blackAgent.name if winner == game.PLAYERS["black"] else whiteAgent.name
	loser_name =  whiteAgent.name if winner == game.PLAYERS["black"] else blackAgent.name
	print(("Game over, " + winner_name+ " ("+game.PLAYER_STR[winner]+") " + "wins against "+loser_name))
	print(game)
	print((" ".join(moves)))
	return winner


parser = argparse.ArgumentParser(description="Run tournament against mohex and output results.")
parser.add_argument("--time", "-t", type=int, help="total time allowed for gitkeach move in seconds.")
parser.add_argument("--boardsize", "-b", type=int, help="width of board to play on.")
parser.add_argument("--verbose", "-v", dest="verbose", action='store_const',
					const=True, default=False,
					help="print board after each move.")
parser.add_argument("--all", "-a", dest="all_openings", action='store_const',
					const=True, default=False,
					help="run games over every board opening.")
args = parser.parse_args()

print("Starting tournament...")
num_games = 169
if(args.time):
	time = args.time
else:
	time = 5
if(args.boardsize):
	boardsize = args.boardsize
else:
	boardsize = 13
agents = [allInputsAgent(), twoInputsAgent(), sixInputsAgent(), fourteenInputsAgent()]
winArray = []
for i in range(4):
	win = []
	for j in range(4):
		if(not i == j):	
			agent= agents[i]
			agent1 = agents[j]
			player1_wins = 0
			player2_wins = 0
			player1_black_wins = 0
			player1_white_wins = 0
			player2_black_wins = 0
			player2_white_wins = 0
			if(args.all_openings):
				for game in range(num_games):
					for x in range(boardsize):
						for y in range(boardsize):
							opening = chr(ord('a')+x)+str(y+1)
							winner = run_game(agent, agent1, boardsize, args.verbose, opening)
							if(winner == gamestate.PLAYERS["white"]):
								player2_wins += 1
								player2_white_wins +=1
							else:
								player1_wins +=1
								player1_black_wins +=1
							winner = run_game(agent1, agent, boardsize, args.verbose, opening)
							if(winner == gamestate.PLAYERS["black"]):
								player2_wins += 1
								player2_black_wins +=1
							else:
								player1_wins +=1
								player1_white_wins +=1
				num_games = num_games*boardsize*boardsize
			else:
				for game in range(num_games):
					winner = run_game(allInputs1, allInputs, boardsize, args.verbose)
					if(winner == gamestate.PLAYERS["white"]):
						white_wins += 1
					winner = run_game(allInputs, allInputs1, boardsize, args.verbose)
					if(winner == gamestate.PLAYERS["black"]):
						black_wins += 1
			wins = {"player1_wins": player1_wins, "player2_wins": player2_wins, "player1_black_wins":player1_black_wins, "player1_white_wins": player1_white_wins, "player2_black_wins": player2_black_wins, "player2_white_wins": player2_white_wins}
			win[j] = wins
	winArray[i] = win

np.savez("wins.npz",wins = winArray)


