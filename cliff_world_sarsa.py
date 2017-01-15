import matplotlib.pyplot as plt
import numpy as np 
from cliff_world_grid import start_state, end_state, nrows, ncols, next_state_reward

#action value pair
actions = range(4) # 0 left, 1 right, 2 up, 3 down
Q = np.zeros((nrows, ncols, 4))
# policy
policy = np.ones((nrows, ncols))/4.0
#some intializations
epislon = 0.1
epochs = 500
max_ts = 1000000
alpha = 0.1

ts_data = []
reward_data = [0]*(epochs)


for t in range(1 , epochs+1):
	S = start_state
	ts = 0 # time-steps
	#choose and select action A
	if np.random.rand() <= epislon:
		print 'explore'
		A = np.random.randint(0, 4, 1)[0]
	else:
		b = np.argmax(Q,2)
		A = b[S[0]][S[1]]
		

	policy[S[0]][S[1]] = A
	print S,A
	while ts<max_ts and S!=end_state:
		ts += 1
		
		S_, reward = next_state_reward(S, A)
		reward_data[t-1] += reward 
		if np.random.uniform(0,1) <= epislon:
			print 'explore'
			A_ = np.random.randint(0, 4, 1)[0]
		else:
			b = np.argmax(Q,2)
			A_ = b[S[0]][S[1]]
			
		print S_,A_
		policy[S_[0]][S_[1]] = A_
		
		#undiscounted 	update

		if S_ == end_state:
			Q[S[0]][S[1]][A] += alpha*(reward - Q[S[0]][S[1]][A] )
		else:
			Q[S[0]][S[1]][A] += alpha*(reward + Q[S_[0]][S_[1]][A_] - Q[S[0]][S[1]][A] ) 			

		S, A = S_, A_

		if S == end_state:
			print 'episode terminated'
			

	ts_data.append(ts)

#plt.plot(range(1, 1+epochs), ts_data)
#plt.show()
print policy
reward_data_cum = np.cumsum(reward_data, axis=0)
plt.plot(range(1,epochs+1), np.array(reward_data_cum)*1.0/np.cumsum( range(1,epochs+1), axis=0 ) )
plt.show()

#print Q