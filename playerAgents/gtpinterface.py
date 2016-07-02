import sys
from resistanceAgent import resistanceAgent
from networkAgent import networkAgent
from treeNetAgent import treeNetAgent
from gamestate import gamestate
version = 0.1
protocol_version = 2
class gtpinterface:
	AGENTS = {"resistance" : resistanceAgent, "network" : networkAgent, "treeNet" : treeNetAgent}
	"""
	Interface for using go-text-protocol to control the program
	Each implemented GTP command returns a string response for the user, along with
	a boolean indicating success or failure in executing the command.
	The interface contains an agent which decides which moves to make on request
	along with a gamestate which holds the current state of the game.
	"""
	def __init__(self, agent_name ="treeNet"):
		"""
		Initialize the list of available commands, binding appropriate names to the
		funcitons defined in this file.
		"""
		commands={}
		commands["name"] = self.gtp_name
		commands["version"] = self.gtp_version
		commands["protocol_version"] = self.gtp_protocol
		commands["known_command"] = self.gtp_known
		commands["list_commands"] = self.gtp_list
		commands["quit"] = self.gtp_quit
		commands["boardsize"] = self.gtp_boardsize
		commands["size"] = self.gtp_boardsize
		commands["clear_board"] = self.gtp_clear
		commands["play"] = self.gtp_play
		commands["undo"] = self.gtp_undo
		commands["genmove"] = self.gtp_genmove
		commands["showboard"] = self.gtp_show
		commands["print"] = self.gtp_show
		commands["set_time"] = self.gtp_time
		commands["winner"] = self.gtp_winner
		commands["hexgui-analyze_commands"] = self.gtp_analyze
		commands["agent"] = self.gtp_agent
		self.commands = commands
		self.game = gamestate(13)
		self.agent_name = agent_name
		try:
			self.agent = self.AGENTS[agent_name]()
		except KeyError:
			print("Unknown agent defaulting to basic")
			self.agent_name = "basic"
			self.agent = self.AGENTS[agent_name]()
		self.agent.set_gamestate(self.game)
		self.move_time = 10
		self.history = []
		register = getattr(self.agent, "register", None)
		if callable(register):
			register(self)

	def send_command(self, command):
		"""
		Parse the given command into a function name and arguments, execute it
		then return the response.
		"""
		parsed_command = command.split()
		#first word specifies function to call, the rest are args
		name = parsed_command[0]
		args = parsed_command[1:]
		if(name in self.commands):
			return self.commands[name](args)
		else:
			return (False, "Unrecognized command")

	def register_command(self, name, command):
		"""
		Add a new command to the commands list under name which
		calls the function command, will also overwrite old commands.
		"""
		self.commands[name] = command

	def gtp_name(self, args):
		"""
		Return the name of the program.
		"""
		return (True, "Neurohex")

	def gtp_version(self, args):
		"""
		Return the version of the program.
		"""
		return (True, str(version))

	def gtp_protocol():
		"""
		Return the version of GTP used.
		"""
		return(True, str(protocol_version))

	def gtp_known(self, args):
		"""
		Return a boolean indicating whether the passed command name is a known command.
		"""
		if(len(args)<1):
			return (False, "Not enough arguments")
		if(args[0] in self.commands):
			return (True, "true")
		else:
			return (True, "false")

	def gtp_list(self, args):
		"""
		Return a list of all known command names.
		"""
		ret=''
		for command in self.commands:
			ret+='\n'+command
		return (True, ret)

	def gtp_quit(self, args):
		"""
		Exit the program.
		"""
		sys.exit()

	def gtp_boardsize(self, args):
		"""
		Set the size of the game board (will also clear the board).
		"""
		if(len(args)<1):
			return (False, "Not enough arguments")
		try:
			size = int(args[0])
		except ValueError:
			return (False, "Argument is not a valid size")
		if size<1:
			return (False, "Argument is not a valid size")
		
		self.game = gamestate(size)
		self.agent.set_gamestate(self.game)
		self.history = []
		return (True, "")

	def gtp_clear(self, args):
		"""
		Clear the game board.
		"""
		self.game = gamestate(self.game.size)
		self.agent.set_gamestate(self.game)
		self.history = []
		return (True, "")

	def gtp_undo(self,args):
		"Undo the last move."
		try:
			self.history.pop()
			self.game = gamestate(self.game.size)
			for (move, player) in self.history:
				self.game.set_turn(player)
				self.game.play(move)
			self.agent.set_gamestate(self.game)
		except ValueError:
			print self.history
			return(False, "Undo failed!")
		return (True, "")

	def gtp_play(self, args):
		"""
		Play a stone of a given colour in a given cell.
		1st arg = colour (white/w or black/b)
		2nd arg = cell (i.e. g5)

		Note: play order is not enforced but out of order turns will cause the
		search tree to be reset
		"""
		if(len(args)<2):
			return (False, "Not enough arguments")
		try:
			x =	ord(args[1][0].lower())-ord('a')
			y = int(args[1][1:])-1

			if(x<0 or y<0 or x>=self.game.size or y>=self.game.size):
				return (False, "Cell out of bounds")

			if args[0][0].lower() == 'w':
				if self.game.turn() == gamestate.PLAYERS["white"]:
					self.game.play((x,y))
					self.agent.move((x,y))
					self.history.append(((x,y), gamestate.PLAYERS["white"]))
					return (True, "")
				else:
					self.game.place_white((x,y))
					self.agent.set_gamestate(self.game)
					self.history.append(((x,y), gamestate.PLAYERS["white"]))
					return (True, "")

			elif args[0][0].lower() == 'b':
				if self.game.turn() == gamestate.PLAYERS["black"]:
					self.game.play((x,y))
					self.agent.move((x,y))
					self.history.append(((x,y), gamestate.PLAYERS["black"]))
					return (True, "")
				else:
					self.game.place_black((x,y))
					self.agent.set_gamestate(self.game)
					self.history.append(((x,y), gamestate.PLAYERS["black"]))
					return (True, "")

			else:
				return(False, "Player not recognized")

		except ValueError:
			return (False, "Malformed arguments")

	def gtp_genmove(self, args):
		"""
		Allow the agent to play a stone of the given colour (white/w or black/b)
		
		Note: play order is not enforced but out of order turns will cause the
		agents search tree to be reset
		"""
		#if user specifies a player generate the appropriate move
		#otherwise just go with the current turn
		if(len(args)>0):
			if args[0][0].lower() == 'w':
				if self.game.turn() != gamestate.PLAYERS["white"]:
					self.game.set_turn(gamestate.PLAYERS["white"])
					self.agent.set_gamestate(self.game)

			elif args[0][0].lower() == 'b':
				if self.game.turn() != gamestate.PLAYERS["black"]:
					self.game.set_turn(gamestate.PLAYERS["black"])
					self.agent.set_gamestate(self.game)
			else:
				return (False, "Player not recognized")

		self.agent.search(self.move_time)
		move = self.agent.best_move()

		if(self.game.winner()):
			return (False, "The game is already over")
		try:
			self.game.play(move)
		except ValueError:
			return (False, "Malformed arguments")
		self.agent.move(move)
		self.history.append((move, gamestate.OPPONENT[self.game.turn()]))
		return (True, chr(ord('a')+move[0])+str(move[1]+1))

	def gtp_time(self, args):
		"""
		Change the time per move allocated to the search agent (in units of secounds)
		"""
		if(len(args)<1):
			return (False, "Not enough arguments")
		try:
			time = int(args[0])
		except ValueError:
			return (False, "Argument is not a valid time limit")
		if time<1:
			return (False, "Argument is not a valid time limit")
		self.move_time = time
		return (True, "")

	def gtp_show(self, args):
		"""
		Return an ascii representation of the current state of the game board.
		"""
		return (True, str(self.game))

	def gtp_winner(self, args):
		"""
		Return the winner of the current game (black or white), none if undecided.
		"""
		if(self.game.winner()==gamestate.PLAYERS["white"]):
			return (True, "white")
		elif(self.game.winner()==gamestate.PLAYERS["black"]):
			return (True, "black")
		else:
			return (True, "none")

	def gtp_analyze(self, args):
		"""
		Added to avoid crashing with gui but not yet implemented.
		"""
		return (True, "")

	def gtp_agent(self, args):
		"Change which agent is used by the player between available options."

		if len(args)<1:
			ret="Available agents:"
			for agent in self.AGENTS.keys():
				if self.agent_name == agent:
					ret+="\n\033[92m"+agent+"\033[0m"
				else:
					ret+="\n"+agent
			return (True, ret)
		else:
			try:
				self.agent = self.AGENTS[args[0]]()
				self.agent.set_gamestate(self.game)
			except KeyError:
				return (False, "Unknown agent")
			self.agent_name = args[0]
			return (True, "")