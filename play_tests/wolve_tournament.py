import argparse
from program import Program
import threading
import time
from gamestate import gamestate
import sys

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

def run_game(blackAgent, whiteAgent, boardsize, verbose = False):
	game = gamestate(boardsize)
	winner = None
	moves = []
	blackAgent.sendCommand("clear_board")
	whiteAgent.sendCommand("clear_board")
	while(True):
		move = blackAgent.sendCommand("genmove black").strip()
		if( move == "resign"):
			winner = game.PLAYERS["white"]
			return winner
		moves.append(move)
		game.place_black(move_to_cell(move))
		whiteAgent.sendCommand("play black "+move)
		if verbose:
			print(blackAgent.name+" v.s. "+whiteAgent.name)
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
		sys.stdout.flush()
		move = whiteAgent.sendCommand("genmove white").strip()
                if( move == "resign"):
                        winner = game.PLAYERS["black"] 
	                return winner
		moves.append(move)
		game.place_white(move_to_cell(move))
		blackAgent.sendCommand("play white "+move)
		if verbose:
			print(blackAgent.name+" v.s. "+whiteAgent.name)
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
		sys.stdout.flush()
	winner_name = blackAgent.name if winner == game.PLAYERS["black"] else whiteAgent.name
	loser_name =  whiteAgent.name if winner == game.PLAYERS["black"] else blackAgent.name
	print("Game over, " + winner_name+ " ("+game.PLAYER_STR[winner]+") " + "wins against "+loser_name)
	print(game)
	print(" ".join(moves))
	return winner

wolve_exe = "/cshome/kjyoung/Summer_2015/benzene-vanilla/src/wolve/wolve 2>/dev/null"
neurohex_exe = "/cshome/kjyoung/Summer_2015/Neurohex/playerAgents/program.py 2>/dev/null"

parser = argparse.ArgumentParser(description="Run tournament against mohex and output results.")
parser.add_argument("num_games", type=int, help="number of *pairs* of games (one as black, one as white) to play between each pair of agents.")
parser.add_argument("--time", "-t", type=int, help="total time allowed for gitkeach move in seconds.")
parser.add_argument("--verbose", "-v", dest="verbose", action='store_const',
					const=True, default=False,
					help="print board after each move.")

args = parser.parse_args()

print("Starting tournament...")
wolve = agent(wolve_exe)
num_games = args.num_games
if(args.time):
	time = args.time
else:
	time = 5
wolve.sendCommand("param_wolve max_time "+str(time))
neurohex = agent(neurohex_exe)
white_wins = 0
black_wins = 0
for game in range(num_games):
	wolve.reconnect()
	wolve.sendCommand("param_wolve max_time "+str(time))
	winner = run_game(wolve, neurohex, 13, args.verbose)
	if(winner == gamestate.PLAYERS["white"]):
		white_wins += 1
	winner = run_game(neurohex, wolve, 13, args.verbose)
	if(winner == gamestate.PLAYERS["black"]):
		black_wins += 1

print "win_rate as white: "+str(white_wins/float(num_games)*100)[0:5]+"%"
print "win_rate as black: "+str(black_wins/float(num_games)*100)[0:5]+"%"



