class unionfind:
	"""
	Unionfind data structure specialized for finding hex connections.
	Implementation inspired by UAlberta CMPUT 275 2015 class notes.
	"""
	def __init__(self):
		"""
		Initialize parent and rank as empty dictionarys, we will
		lazily add items as nessesary.
		"""
		self.parent = {}
		self.rank = {}

	def join(self, x, y):
		"""
		Merge the groups of x and y if they were not already,
		return False if they were already merged, true otherwise
		"""
		rep_x = self.find(x)
		rep_y = self.find(y)

		if rep_x == rep_y:
			return False
		if self.rank[rep_x] < self.rank[rep_y]:
			self.parent[rep_x] = rep_y
		elif self.rank[rep_x] >self.rank[rep_y]:
			self.parent[rep_y] = rep_x
		else:
			self.parent[rep_x] = rep_y
			self.rank[rep_y] += 1
		return True

	def find(self, x):
		"""
		Get the representative element associated with the set in
		which element x resides. Uses grandparent compression to compression
		the tree on each find operation so that future find operations are faster.
		"""
		if x not in self.parent:
			self.parent[x] = x
			self.rank[x] = 0

		px = self.parent[x]
		if x == px: return x

		gx = self.parent[px]
		if gx==px: return px

		self.parent[x] = gx

		return self.find(gx)

	def connected(self, x, y):
		"""
		Check if two elements are in the same group.
		"""
		return self.find(x)==self.find(y)
