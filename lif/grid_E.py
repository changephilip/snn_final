import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import scipy.interpolate
import scipy as sp

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)

T = np.loadtxt("gride2.txt",dtype=np.float)
# Make data.
X = T[:,0:1].flatten()
Y = T[:,1:2].flatten()
Z = T[:,2:3].flatten()

x = np.linspace(min(X),max(X),num=100)
y = np.linspace(min(Y),max(Y),num=50)
x,y = np.meshgrid(x,y)

from scipy.interpolate import LinearNDInterpolator
interp = LinearNDInterpolator(list(zip(X,Y)),Z)

z = interp(x,y)
#plt.pcolormesh(x,y,z,shading='auto')

sX = x.flatten()
sY = y.flatten()
sZ = z.flatten()


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.xlabel("I/mA")
plt.ylabel("sigma")
#zlabel("E(ISI)/ms")

plt.title("E(ISI) - (I,sigma)")
#plt.show()
