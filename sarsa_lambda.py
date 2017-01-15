import numpy as np
import matplotlib.pylab as plt
############ making a grid  ########################
alpha = 0.2
m, n = 4, 12
grid = -1*np.ones((m, n))
epsilon = 0.9
trace_param = 0.01
# terminal states
for i in range(n):
	if i==0:
		grid[0][i] = 0
	elif i==11:
		grid[0][i] = 1
	else:
		grid[0][i]  = -100
		grid[3][i] = -100

#grid[2][3:7] = -100
#grid[1][1:2] = -100
#grid[1][9:11] = -100
print grid


convertAction = {'left':0, 'riht':1, 'upup':2, 'down':3}
getActionString = {0:'left', 1:'riht', 2:'upup', 3:'down'}
startState = (0, 0)
terminalState = (0, n-1)

def displayPolicy():
	for i in range(m):
		for j in range(n):
			print getAction((i,j), 0.0),
		print ''

def getActionSet(state, m=4, n=12):

	x, y = state
	if x==0 and y==0:
		return ("down", 'riht')
	if x==0 and y==n-1:
		return ('down', 'left')
	if x==m-1 and y==0:
		return ('upup', 'riht')
	if x==m-1 and y==n-1:
		return ('upup', 'left')

	if x==0:
		return ('down', 'riht', 'left')
	elif x==m-1:
		return ('upup', 'riht', 'left')
	elif y==0:
		return ('upup', 'down', 'riht')
	elif y==n-1:
		return ('upup', 'down', 'left')
	else:
		return ('upup', 'down', 'riht', 'left')

def nextState(state, action):
	x, y = state
	if action == 'riht':
		return (x,y+1)
	elif action == 'left':
		return (x,y-1)
	elif action == 'upup':
		return (x-1, y)
	else:
		return (x+1, y)





##############    Agent methods     ########################
# making of the Q table #

Q = np.zeros((m, n, 4))
greedycount = 0
def getAction(state, epsilon):
	''' returns the string showing the action.
	'''
	global greedycount
	actions = getActionSet(state, m, n)
	if epsilon_greedy(epsilon):
		#print 'here'
		greedycount += 1
		#print np.random.randint(0,len(actions),1)
		return actions[np.random.randint(0,len(actions),1)[0]]

	else:
		# greedy policy
		q_max, Action = -1e5, ''
		#print Q[state[0]][state[1]]
		for action in actions:
			#print action
			if q_max < Q[state[0]][state[1]][convertAction[action]]:
				q_max = Q[state[0]][state[1]][convertAction[action]]
				Action = action
		
		return Action

def epsilon_greedy(epsilon):
	
	
	return np.random.uniform(0, 1, 1) < epsilon



#################################################
 # sarsa #

numEpisodes = 1000
maxSteps = 5000
requiredSteps = []
just = []
firstattempt = 0
gotfailure = []
for i in range(numEpisodes):

	Z = np.zeros((m, n, 4))

	epsilon = 1.0/(i+1)
	just = []
	state = startState
	action = getAction(state, epsilon)	

	steps = 0
	firstattempt = 0
	
	while state!=terminalState and steps < maxSteps:
		just.append(action)
		next_state = nextState(state, action)
		R = grid[state[0]][state[1]]
		if R==-100: 
			firstattempt +=1
			next_state = startState
		next_action  =  getAction(next_state, epsilon)
		# undiscounted
		delta=R+(Q[next_state[0]][next_state[1]][convertAction[next_action]]-Q[state[0]][state[1]][convertAction[action]] )
																		
		Z[state[0]][state[1]][convertAction[action]] += 1



		Q += alpha*delta*Z
		Z *= trace_param*1.0
		state = next_state
		action = next_action
		steps += 1
	
	requiredSteps.append(steps)
	gotfailure.append(firstattempt)
	#print steps
	if i==0:
		displayPolicy()
	





#print requiredSteps
print '#####################################'
print gotfailure[0]
print '#####################################'
print requiredSteps[0]
print '#####################################'
displayPolicy()
#plt.plot(range(1,len(requiredSteps)+1), requiredSteps)
#plt.show()

plt.plot(range(1,len(requiredSteps)+1), requiredSteps,'r', range(1,len(requiredSteps)+1),  gotfailure, 'b')
plt.show()

print gotfailure

#print Q

