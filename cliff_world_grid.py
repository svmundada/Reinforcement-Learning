#code for the grid 
import numpy as np 

nrows = 4
ncols = 12

grid = [[-1]*ncols for _ in range(nrows)]

for i in range(1,ncols-1):
	grid[nrows-1][i] = -100

start_state = (nrows-1, 0)
end_state = (nrows-1, ncols-1)

def possible_actions(S):
	'''Will never accept terminal state # 0 left, 1 right, 2 up, 3 down'''
	i, j = S[0], S[1]
	if i==0 and j==0:
		return (1, 3)
	elif i==0 and j==ncols-1:
		return(0, 3)
	elif j==0 and i==nrows-1:
		return (1, 2)
	elif i==0:
		return (0, 1, 3)
	elif j==0:
		return (1, 2, 3)
	elif j==ncols-1:
		return (0, 2, 3)
	elif i==nrows-1:
		return (0, 1, 2)
	else:
		return (0, 1, 2, 3)


def next_state(S,A):
	i, j = S[0], S[1]
	if A==0: #left
		return (i, j-1)
	elif A==1:
		return (i ,j+1)
	elif A==2:
		return (i-1, j)
	else:
		return (i+1, j)

def correction_off_grid(S):
	i, j = S[0], S[1]
	if i<0:
		i=0
	elif j<0:
		j = 0
	elif j==ncols:
		j = ncols-1
	elif i==nrows:
		i = nrows-1

	return (i, j)

def next_state_reward(S, A):
	global grid
	S_ = correction_off_grid( next_state(S,A) )
	i, j = S_[0], S_[1]
	reward = None
	if grid[i][j] == -100:
		reward = -100
		S_ = start_state
	elif i==nrows-1 and j==ncols-1:
		reward = 0
	else:
		reward = -1

	return S_, reward
