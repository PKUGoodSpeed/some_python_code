__author__ = 'zeboli1'

import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal

delta = 0.05
b = 2.5
radi = 4.0

Low = -0.03
High = 0.03

x = y = np.arange(-50.0,50.0,delta)
X,Y = np.meshgrid(x,y)

def volu_strain(x,y):
    r = np.sqrt(x**2 + y**2)
    cta = np.arctan2(y,x)
    y = -b*np.sin(cta)/r/4./np.pi
    return y

def shear_strain(x,y):
    r = np.sqrt(x**2 + y**2)
    cta = np.arctan2(y,x)
    return 3.*b*(np.cos(cta) + np.cos(3.0*cta))/16./np.pi/r

def tetra_strain(x,y):
    r = np.sqrt(x**2 + y**2)
    cta = np.arctan2(y,x)
    return -1.*b*np.cos(2.0*cta)/8./np.pi/r + -1.*b*np.cos(0.00000001*cta)/8./np.pi/r


origin = 'lower'
Z = tetra_strain(X,Y)
#interior = np.sqrt((X**2 +Y**2)) < 6.7
#Z[interior] = np.ma.masked
#Z[interior] = np.nan
CS = plt.contourf(X, Y, Z, 10,locator=ticker.LogLocator(), cmap=cm.coolwarm,
                        origin=origin)
def f(x,n):
    return pow(x,1./n)

def f_inverse(x,n):
    return pow(x,n*1.)


lev_exp = np.arange(Low,High,(High-Low)/20.)

cmap = plt.cm.get_cmap("coolwarm")
cmap.set_under(cmap(0))
cmap.set_over(cmap(400))

CS3 = plt.contourf(X, Y, Z, lev_exp ,
                   cmap = cmap,
                   extend='both',
                   origin=origin)


plt.title('volumetric strain $\epsilon_bb$')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(CS3)
plt.show()