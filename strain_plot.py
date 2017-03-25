__author__ = 'zeboli1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal

font_size = 14

width = 17
height = 7.5

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

extends = ["volumetric strain $\epsilon_V$", "shear strain $\epsilon_{bn}$", "tetragonal strain $\epsilon_{bb}$"]
levs = np.arange(Low,High,(High-Low)/15.)
cmap = plt.cm.get_cmap("coolwarm")
cmap.set_under(cmap(0))
cmap.set_over(cmap(400))


fig, axs = plt.subplots(1, 3,figsize=(width, height))
Z0 = volu_strain(X,Y)
Z1 = shear_strain(X,Y)
Z2 = tetra_strain(X,Y)

cs1 = axs[0].contourf(X, Y, Z0, levs, cmap=cmap, extend="both", origin=origin)
axs[0].set_title("$\epsilon_V$",family = 'serif',size = 32, y=1.04)
axs[0].set_xticks([-40,-20,0,20,40])
axs[0].set_xticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[0].set_xlabel("(a)",family = 'serif', size = 25, y = 1.4)
axs[0].set_yticks([-40,-20,0,20,40])
axs[0].set_yticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[0].set_aspect('equal')

#axs[0].set_xlabel("(a)",family = 'serif',weight = 'bold',size = font_size)
#axs[0].set_xticks([-50,0,50],family = 'serif',weight = 'bold',size = font_size)
#axs[0].set_yticks([-50,0,50])

cs2 = axs[1].contourf(X, Y, Z1, levs, cmap=cmap, extend="both", origin=origin)
axs[1].set_title("$\epsilon_{bn}$",family = 'serif',size = 32, y=1.05)
axs[1].set_xticks([-40,-20,0,20,40])
axs[1].set_xticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[1].set_xlabel("(b)",family = 'serif', size = 25, y = 1.4)
axs[1].set_yticks([-40,-20,0,20,40])
axs[1].set_yticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[1].set_aspect('equal')

cs3 = axs[2].contourf(X, Y, Z2, levs, cmap=cmap, extend="both", origin=origin)
axs[2].set_title("$\epsilon_{bb}$",family = 'serif',size = 32, y=1.05)
axs[2].set_xticks([-40,-20,0,20,40])
axs[2].set_xticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[2].set_xlabel("(c)",family = 'serif', size = 25, y = 1.4)
axs[2].set_yticks([-40,-20,0,20,40])
axs[2].set_yticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
axs[2].set_aspect('equal')

vmin,vmax = cs3.get_clim()
#-- Defining a normalised scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax3 = fig.add_axes([0.9, 0.1, 0.017, 0.8])
#-- Plotting the colormap in the created axes
cb1 = mpl.colorbar.ColorbarBase(ax3,cmap=cmap, norm=cNorm , extend = "both")
#ax3.set_yticks([0.1,0.25,0.5,0.9])
#ax3.set_yticklabels(["-0.03","-0.1","0.1","0.03"],rotation=0,family = 'serif',size = 24)
ax3.get_yaxis().set_ticks([])
for j, lab in enumerate(['$-0.03$','$-0.01$','$0.01$','$0.03$']):
    ax3.text(2.5, j/3.0*19./20. + 1./40, lab,
             ha='center',
             va='center',
             family = 'serif',size = 22)
#cbar.ax.get_yaxis().labelpad = 15


fig.subplots_adjust(left=0.05,right=0.85)
plt.show()

'''
plt.title('volumetric strain $\epsilon_bb$')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(CS3)
'''
plt.show()
