import numpy as np
import pandas as pd
import time
import sys

def retCord(something):
    return np.array(list(zip(something[0],something[1])))

def readMaze():
    H=49
    W=65
    import linecache
    t=linecache.getlines('maze.csv')

    npMaze=np.zeros((H,W),dtype=int)
    
    for j in range(0,H):
        thisRow = t[j].split()[0]
        for i in range (0,W):
            npMaze[j,i]=thisRow[i]
    wall = np.where(npMaze==0)
    road = np.where(npMaze==1)
    target = np.where(npMaze==2)
    return npMaze,retCord(wall),retCord(road),retCord(target)

def randomStart(road):
    import random
    n = random.randint(0,len(road))
    return road[n]

class Maze():
    def __init__(self):
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)

        self.H=49
        self.W=65
        
        import linecache
        t=linecache.getlines('maze.csv')

        self.npMaze=np.zeros((self.H,self.W),dtype=int)
    
        for j in range(0,self.H):
            thisRow = t[j].split()[0]
            for i in range (0,self.W):
                self.npMaze[j,i]=thisRow[i]
        self.wall = retCord(np.where(self.npMaze==0))
        self.road = retCord(np.where(self.npMaze==1))
        self.terminal = retCord(np.where(self.npMaze==2))[0]

        print(self.terminal)

        self.s = randomStart(self.road)
        #self.s=np.array([37,60])
        #print(self.npMaze[37][60])
        


    def eq(self,pointA,pointB):
        if pointA[0] == pointB[0] and pointA[1]==pointB[1]:
            return True
        else:
            return False
             
    def checkBoundary(self,point):
        if point[0] >=  self.H or point[0] <=0:
            return False
        if point[1] >= self.W or point[1] <=0:
            return False
        return True

    def reset(self):
        return self.s

    def step(self, action):
        if action == 0 and self.s[0]!= 0:   # up
            s_ = self.s + np.array([-1,0])
        elif action == 1 and self.s[0]!=48:   # down
            s_ = self.s + np.array([1,0])
        elif action == 2 and self.s[1]!=0 :   # left
            s_ = self.s + np.array([0,-1])
        elif action == 3 and self.s[1]!=64:   # right
            s_ = self.s + np.array([0,1])
        
        # reward function
        if self.eq(s_,self.terminal):
            reward = 100
            done = True
            s_ = 'terminal'
        elif self.npMaze[s_[0]][s_[1]]==0:
            reward = -10
            done = False
            s_ = self.s
        else:
            reward = -0.1
            done = False
            self.s = s_

        #print("return:",s_)
        return s_, reward, done

    def render(self):
        time.sleep(0.1)




class QLearningTable:
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.6):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

def update():
    # 跟着行为轨迹
    df = pd.DataFrame(columns=('state','action_space','reward','Q','action'))
    iter = 0
    flag = False
    for episode in range(1):
        # initial observation
        observation = env.s
        while iter < 100000:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            q = RL.q_table.loc[str(observation),action]
            df = df.append(pd.DataFrame({'state':[observation],'action_space':[env.action_space[action]],'reward':[reward],'Q':[q],'action':action}), ignore_index=True)

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                flag=True

                break
            if reward!=-10:
                iter=iter+1

    # end of game
    print('game over')
    if flag:
        print("Find The Path To Exit\n")
    else:
        print("No Path to Exit")
    df.to_csv('action.csv')
    RL.q_table.to_csv('q_table.csv')
    return df,flag
    #env.destroy()

import networkx as nx
def plotPath(env:Maze,T):
    tt = T[T['reward']==-0.1]['state']
    nodeList=[]
    for node in tt:
        if tuple(node) not in nodeList:
            x=tuple(node)[0]
            y=tuple(node)[1]
            if env.npMaze[x][y]==1:
                nodeList.append(tuple(node))
    
    N=len(nodeList)
    G = nx.Graph()
    for n in range(0,N):
        G.add_node(n)

    for i in range(0,N):
        for j in range(i+1,N):
            pA=nodeList[i]
            pB=nodeList[j]
            if np.abs(pA[0]-pB[0])==1 and np.abs(pA[1]==pB[1]):
                G.add_edge(i,j)
            if np.abs(pA[1]-pB[1])==1 and np.abs(pA[0]==pB[0]):
                G.add_edge(i,j)

    path = nx.dijkstra_path(G,0,N-1)
    pMaze=env.npMaze.copy()
    for n in path[1:]:
        i=nodeList[n][0]
        j=nodeList[n][1]
        pMaze[i][j]=3
    
    
    pMaze[nodeList[0][0]][nodeList[0][1]]=4
    import matplotlib.pyplot as plt
    plt.imshow(pMaze)
    plt.show()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    T,flag=update()
    
    print(RL.q_table)
    if flag:
        plotPath(env,T)