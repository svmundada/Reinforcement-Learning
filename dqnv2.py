import tensorflow as tf
import numpy as np
import gym
import copy
env = gym.make("Cartpole-v0")




#env.monitor.start('./cartpole-experiment-2')
class dqn():
    
    def __init__(self, sess, d):
        
        self.sess = sess
        
        self.hidden_dim = len(d) -1 
        self.ACTIONS = d[-1]
        self.GAMMA = 0.99
        self.weights = [None]*(self.hidden_dim)
        self.biases = [None]*(self.hidden_dim)
        self.layers = [None]*(self.hidden_dim + 1)
        self.out = None
        self.loss = None
        self.loss1 = None
        self.loss2 = None
        self.td_error = None
        self.optimizer = None
        self.Qsa = None
        
        
        self.state = tf.placeholder(dtype=tf.float32, shape=((None, d[0])), name="state")
        self.action = tf.placeholder(dtype=tf.float32, shape=((None, self.ACTIONS)), name="actions")
        self.target = tf.placeholder(dtype=tf.float32, shape=(None), name="target")     
        self.clipped = tf.placeholder(dtype=tf.float32, shape=(None), name="clipped")

        self.learning_rate = 1e-3
                                       
        self.makeParameters(d)
        self.createNetwork()
        self.buildQsa()
        self.train(self.learning_rate)
                        
            
        self.sess.run(tf.initialize_all_variables()) 
     
    
    def setClipped(self, feed):

        td_error_value = self.sess.run(self.td_error, feed)

        clipped_value = np.zeros_like(td_error_value)

        clipped_value[np.logical_and(td_error_value>-1, td_error_value<1)] = 1.0

        return clipped_value



    def getTarget(self, state_value, reward, terminal):
        '''
            r + gamma*Qs'max if not terminal and r if terminal
            only used when training.
            so except state others can be real value i.e numpy array
            and state is a placeholder for s'
        '''
        
        #print '########',terminal, reward, self.getQmax(state_value)
        #print ""
        #print terminal.shape, self.getQmax(state_value).reshape((-1, 1)).shape
        return reward + self.GAMMA*np.multiply(self.getQmax(state_value).reshape((-1, 1)), terminal)
        
        
                                       
    def train_step(self, target_value, state_value, action_value):
        
        #print self.Qsa
        #print "getQsa", self.getQsa(state_value, action_value)
        #print self.loss
        feed = {self.state:state_value, self.action:action_value, self.target:target_value}
        clipping = self.setClipped(feed) # sets the clipping_bool as true for have td_error in (-1, 1)
        feed[self.clipped] = clipping
        
        return self.sess.run([self.loss, self.optimizer], feed_dict=feed)

                                       
    def train(self, learning_rate):
        
        self.td_error = self.target - self.Qsa
        self.loss2 = tf.abs(self.td_error)/1e5
        self.loss1 =  tf.square(self.target - self.Qsa)

        self.loss = tf.reduce_sum( self.loss1 + tf.mul(self.clipped, self.loss1 - self.loss2), 0) /1e5

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                                           
    
    def getQsa(self, state_value, action_value):
        return self.sess.run(self.Qsa, feed_dict={self.state:state_value, self.action:action_value})                             
                                       
    def buildQsa(self):

        self.Qsa = tf.reshape( tf.reduce_sum(tf.mul(self.out, self.action), reduction_indices=1), (-1, 1) )                        
                                       
    def makeParameters(self, d):
        for i in range(self.hidden_dim):
            self.weights[i] = tf.Variable(tf.random_uniform((d[i],d[i+1]), -1, 1))
            self.biases[i] = tf.Variable(tf.random_uniform((d[i+1], ), -1, 1))
        
        
    def createNetwork(self):
        self.layers[0] = self.state
        for i in range(1, self.hidden_dim): 
            temp = tf.matmul(self.layers[i-1], self.weights[i-1])
            self.layers[i] = tf.nn.relu(tf.nn.bias_add(temp, self.biases[i-1]))
            
        self.layers[-1] = tf.nn.bias_add(tf.matmul(self.layers[-2], self.weights[-1]), self.biases[-1])
        self.out = self.layers[-1]
            
    def updateParams(self, weights_value, biases_value):

        updatew = [ self.weights[i].assign(weights_value[i]) for i in range(len(weights_value)) ]
        updateb = [ self.biases[i].assign(biases_value[i]) for i in range(len(biases_value)) ]

        self.sess.run([updatew, updateb])
        



    def getQ(self, state_value ):
        return self.sess.run(self.out, feed_dict={self.state:state_value})
    
    def getQmax(self, state_value):
        #print self.getQ(state_value)
        return np.amax(self.getQ(state_value), 1)
    
    def getMaxAction(self, state_value):
        Q = self.getQ(state_value)
        return np.argmax(Q, 1)
    
    def getParams(self):
        w = self.sess.run(self.weights)
        b = self.sess.run(self.biases)
        #print "weights", len(w)
        return (w, b)
    
class ExperienceReplay():

    def __init__(self, memory_size, state_size, action_size):
        self.size = memory_size
        self.state1 = np.zeros((self.size, state_size))
        self.state2 = np.zeros((self.size, state_size))
        self.action = np.zeros((self.size, action_size))
        self.terminal = np.zeros((self.size, 1))
        self.reward = np.zeros((self.size, 1))
        self.count = 0
        
    def getBatch(self, sample_size):
        '''
        will return batch
        '''
        if self.size > sample_size:
            return np.random.choice(self.count, self.count)
        else:
            return np.random.choice(self.size, sample_size, replace=False)
        

    def getSample(self, sample_size):
        batch = self.getBatch(sample_size)
        #print 'batch', batch
        return self.state1[batch], self.state2[batch], self.action[batch], self.reward[batch], self.terminal[batch]

    def checknRemove(self):
        if self.count >= self.size:
            np.delete(self.state1, 0, 0)
            np.delete(self.state2, 0, 0)
            np.delete(self.action, 0, 0)
            np.delete(self.reward, 0, 0)
            np.delete(self.terminal, 0, 0)
            self.count -= 1
            
    def add(self, s1, s2, a, r, t):
        
        self.checknRemove()
        
    
        self.state1[self.count] = s1
        self.state2[self.count] = s2
        self.terminal[self.count] = t
        self.action[self.count] = a
        self.reward[self.count] = r
        
        self.count += 1
        



OBSERVATION  = env.observation_space.shape[0]
ACTIONS = env.action_space.n
EPISODES  = 1000
BATCHSIZE = 500
MEMORY_SIZE = 5000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.025
UPDATE_PARAMS = 100
MAXSTEPS = 60
0#LEARNABLE = 150

sess = tf.Session()

d = [OBSERVATION, 100, 100, ACTIONS]

mydqn = dqn(sess, d)
target_dqn = dqn(sess, d)

#mydqn = dqn(sess, [4, 3, 2])
#target_dqn = dqn(sess, [4, 3, 2])

memory = ExperienceReplay(MEMORY_SIZE, OBSERVATION, ACTIONS)
c = 0
epsilon = INITIAL_EPSILON

eps_count = 0
for epi in range(1, EPISODES+1):
    
    steps = 0
    
    terminal = False
    state1 = env.reset()
    #print 'state1 reseted', state1
    Loss = []
    #for _ in xrange(15):

    if epi % 100:
         mydqn.learning_rate *= 0.99

    
    greedy_count = 0
    while ( (not terminal) and (steps <= MAXSTEPS) ) :
        steps += 1
        
        env.render()

        if eps_count == 20:
            epsilon  *= 0.99
            eps_count = 0
        else:
            eps_count += 1


        action = np.zeros(ACTIONS)
        hot = None
        if np.random.rand() < epsilon:
            hot = env.action_space.sample()
            action[hot] = 1.0
            greedy_count += 1
        else:
            hot  = mydqn.getMaxAction(np.atleast_2d(state1))[0]
            action[hot] = 1.0

        #print 'e greedy action', action

        state2, reward, terminal, _ = env.step(hot)

        # print "state2",state2
        # print "reward",reward
        # print "terminal",terminal

        

        if terminal :
            reward = -500

        if steps >= 500 and terminal:
            print "max reached", terminal, steps
            reward = 1


        memory.add(state1, state2, action, reward, 0 if terminal else 1, )

        
        state1 = copy.deepcopy(state2)


        # training starts
  
       
        states1, states2, actions, rewards, terminals = memory.getSample(BATCHSIZE)

        # print 'states1', states1
        # print 'states2', states2
        # print 'actions' ,actions
        # print 'rewards', rewards
        # print 'terminals', terminals

        #print 'target Q values', target_dqn.getQ(states2)

        target_Q = target_dqn.getTarget(states2, rewards, terminals)
        #print 'targetQ', target_Q
        loss, _ = mydqn.train_step(target_Q, states1, actions)
        
        z = mydqn.getQsa(states1, actions)

        #print "Q values",z 
        #print "loss is", loss
        zq =  (target_Q - z.reshape((-1, 1)))
        #print zq
        zzq = np.mean(zq*zq)
        #print zzq
        Loss.append(loss)

        #print 'state1 is updated', state1

        #assert False

        if c == UPDATE_PARAMS:
            #LEARNABLE += 2
            print 'updated'
            dqnW, dqnB = mydqn.getParams()
            #c, d = target_dqn.getParams()
            #a, b = copy.deepcopy(dqnW), copy.deepcopy(dqnB)
            target_dqn.updateParams(copy.deepcopy(dqnW), copy.deepcopy(dqnB))
            #ddqnW, ddqB = target_dqn.getParams()
            # print "actual", dqnW[0]
            # print ""
            # print "deep copy check", a[0]
            # print ""
            # print "after updating the target", ddqnW[0]
            # print ""
            # print "before updating the target" ,c[0]
            # if id(dqnW) == id(ddqnW):
            #     print "id's are same"
            # if id(dqnW[0]) == id(ddqnW[0]):
            #     print "not a deep copy"
            
            # assert False
            c = 0
        elif c!=UPDATE_PARAMS:
            c += 1


    #assert False
    #break
    
    print "episode",epi," rewards ", steps, "greedy", greedy_count, "epsilon ",epsilon, "loss ", np.mean(Loss),"lr ", mydqn.learning_rate







#env.monitor.close()


