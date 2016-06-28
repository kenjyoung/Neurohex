import argparse
from program import Program
import threading
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

def run_game(blackAgent, whiteAgent, boardsize, verbose = False, opening = None):
	game = gamestate(boardsize)
	winner = None
	moves = []
	blackAgent.sendCommand("clear_board")
	whiteAgent.sendCommand("clear_board")
	if opening:
		game.place_black(move_to_cell(opening))
		blackAgent.sendCommand("play black "+opening)
		whiteAgent.sendCommand("play black "+opening)
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
			return winner
		sys.stdout.flush()

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

mohex_exe = "/cshome/kjyoung/Summer_2015/benzene-vanilla/src/mohex/mohex 2>/dev/null"
neurohex_exe = "/cshome/kjyoung/Summer_2015/Neurohex/playerAgents/program.py 2>/dev/null"

parser = argparse.ArgumentParser(description="Run tournament against mohex and output results.")
parser.add_argument("num_games", type=int, help="number of *pairs* of games (one as black, one as white) to play between each pair of agents.")
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
mohex = agent(mohex_exe)
num_games = args.num_games
if(args.time):
	time = args.time
else:
	time = 5
if(args.boardsize):
	boardsize = args.boardsize
else:
	boardsize = 13
mohex.sendCommand("param_mohex max_time "+str(time))
neurohex = agent(neurohex_exe)
mohex.sendCommand("boardsize "+str(boardsize)+" "+str(boardsize))
neurohex.sendCommand("boardsize "+str(boardsize))
white_wins = 0
black_wins = 0
if(args.all_openings):
	for game in range(num_games):
		for x in range(boardsize):
			for y in range(boardsize):
				opening = chr(ord('a')+x)+str(y+1)
				mohex.reconnect()
				mohex.sendCommand("param_mohex max_time "+str(time))
				mohex.sendCommand("boardsize "+str(boardsize)+" "+str(boardsize))
				winner = run_game(mohex, neurohex, boardsize, args.verbose, opening)
				if(winner == gamestate.PLAYERS["white"]):
					white_wins += 1
				winner = run_game(neurohex, mohex, boardsize, args.verbose, opening)
				if(winner == gamestate.PLAYERS["black"]):
					black_wins += 1
	num_games = num_games*boardsize*boardsize
else:
	for game in range(num_games):
		mohex.reconnect()
		mohex.sendCommand("param_mohex max_time "+str(time))
		mohex.sendCommand("boardsize "+str(boardsize)+" "+str(boardsize))
		winner = run_game(mohex, neurohex, boardsize, args.verbose)
		if(winner == gamestate.PLAYERS["white"]):
			white_wins += 1
		winner = run_game(neurohex, mohex, boardsize, args.verbose)
		if(winner == gamestate.PLAYERS["black"]):
			black_wins += 1


print "win_rate as white: "+str(white_wins/float(num_games)*100)[0:5]+"%"
print "win_rate as black: "+str(black_wins/float(num_games)*100)[0:5]+"%"



