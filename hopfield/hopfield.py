import numpy as np
import matplotlib.pyplot as plt

#输入是记录各个城市之间距离的矩阵
#W，U，V
class HopfieldTSP():
    def __init__(self,cities):
        self.cities = cities
        self.nCities = cities.shape[0]
        self.u0 = 0.02
        self.deltaT = 1e-4
        self.A = 1000
        self.D =10
        self.W = np.zeros((self.nCities,self.nCities))
        self.U = np.zeros((self.nCities,self.nCities))
        self.V = np.zeros((self.nCities,self.nCities))
        self.initWeight()
        self.initNode()

#初始化权重矩阵
    def initWeight(self,):
        for i in range(self.nCities):
            for j in range(self.nCities):
                self.W[i,j] = np.sqrt(np.sum((self.cities[i] - self.cities[j])**2))

    def initNode(self):
        for i in range(self.nCities):
            for j in range(self.nCities):
                self.U[i,j] = 0.5 * self.u0 * np.log(self.nCities - 1) + np.random.random()*2 -1

    def getEnergy(self):
        energy = 0.0
#逐次累加改进后的能量函数
        for i in range(self.nCities):
            #约束每行只有1个1
            energy += 0.5 * self.A * (np.sum(self.V[i, :]) - 1)**2 
            #约束每列只有1个1
            energy += 0.5 * self.A * (np.sum(self.V[:,i]) - 1)**2
            #路程
            for j in range(self.nCities):
                for k in range(self.nCities):
                    t = k + 1 if k + 1 < self.nCities else 0
                    energy += 0.5 * self.D * self.W[i,j] * self.V[i,k] * self.V[j,t]
                
        return energy

    def diff(self,i ,j):
        t = j+1 if j+1 < self.nCities else 0
        return -self.A*(np.sum(self.V[i]) - 1) - self.A * (np.sum(self.V[:,j]) - 1) - self.D * self.W[i,:].dot(self.V[:,t])

    def check(self):
        pos= np.where(self.V<0.2,0,1)
        flag  =True
        if np.sum(pos) != self.nCities:
            flag = False
        for i in range(self.nCities):
            if np.sum(pos[:,i])!=1:
                flag = False
            if np.sum(pos[i,:])!=1:
                flag = False
        return flag

    def __call__(self):
        running_energy = []
        iter = 0
        while not self.check() and iter < 1000000:
            iter += 1
            if iter %10000 ==0 :
                print(iter,running_energy[-1])
            for i in range(self.nCities):
                for j in range(self.nCities):
                    self.U[i,j] += self.deltaT * self.diff(i,j)

            self.V = 0.5 * ( 1+ np.tanh(self.U/ self.u0))
            energy = self.getEnergy()
            running_energy.append(energy)
        return running_energy, np.where(self.V<0.2,0,1), iter


def test():
    cities = np.array([[2,6],[2,4],[1,3],[4,6],[5,5],[4,4],[6,4],[3,2]])
    solver = HopfieldTSP(cities)
    energy, answer,iter = solver()
    print(answer)
    print(iter)
    #print(energy)
    plt.plot(energy)
    plt.show()

#ATT48 is a set of 48 cities (US state capitals) from TSPLIB. The minimal tour has length 33523.     
#https://people.sc.fsu.edu/~jburkardt/datasets/tsp/att48_xy.txt
def testUS48():
    cities=np.loadtxt('att48_xy.txt')
    solver = HopfieldTSP(cities)
    energy, answer,iter = solver()
    print(answer)
    print(iter)
    #print(energy)
    plt.plot(energy)
    plt.show()

if __name__ == "__main__":
    #cities = np.array([[2,6],[2,4],[1,3],[4,6],[5,5],[4,4],[6,4],[3,2]])
    cities=np.loadtxt('att48_xy_8.txt')
    solver = HopfieldTSP(cities)
    energy, answer,iter = solver()
    print(answer)
    print(iter)
    #print(energy)
    plt.plot(energy)
    plt.show()

       