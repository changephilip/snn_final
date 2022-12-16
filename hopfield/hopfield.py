import numpy as np
import matplotlib.pyplot as plt

#输入是记录各个城市之间距离的矩阵
#W，U，V
class HopfieldTSP():
    def __init__(self,cities):
        self.cities = cities
        self.nCitites = cities.shape[0]
        self.u0 = 0.02
        self.deltaT = 1e-4
        self.A = 200
        self.D =100
        self.W = np.zeros((self.n,self.n))
        self.U = np.zeros((self.n,self.n))
        self.V = np.zerors((self.n,self.n))
        self.initWeight()
        self.initNode()

#初始化权重矩阵
    def initWeight(self,):
        for i in range(self.nCitites):
            for j in range(self.nCitites):
                self.W[i,j] = np.sqrt(np.sum((self.cities[i] - self.citites[j])**2))

    def initNode(self):
        for i in range(self.nCitites):
            for j in range(self.n):
                self.U[i,j] = 0.5 * self.u0 * np.log(self.nCitites - 1) + np.random.random()

    def getEnergy(self):
        energy = 0.0
#逐次累加改进后的能量函数
        for i in range(self.nCitites):
            #约束每行只有1个1
            energy += 0.5 * self.A * (np.sum(self.V[i, :] - 1))**2 
            #约束每列只有1个1
            energy += 0.5 * self.A * (np.sum(self.V[i:,1] - 1))**2
            #路程
            for j in range(self.n):
                for k in range(self.n):
                    t = k + 1 if k + 1 < self.nCitites else 0
                    energy += 0.5 * self.D * self.W[i,j] * self.V[i,k] * self.V[j,t]
                
        return energy

    def diff(self,i ,j):
        t = j+1 if j+1 < self.nCitites else 0
        
       