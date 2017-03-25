__author__ = 'zeboli1'

import numpy as np
from scipy import interpolate as itp
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


'''
Using values defined at each discrized sites to generate a linear interpolation function
input sites is an nX2 array storing the coordiantes of sites
'''
class linear_inter_scheme:

    #The coordinates of the discretized sites will not change during the iteration
    def __init__(self,sites):
        self.num_sites = np.shape(sites)[0]
        self.x_ary = sites[:,0]
        self.y_ary = sites[:,1]

    #for extracting boundary values
    def create_1Dmesh(self,radius,n_grid=50):
        self.n_grid1D = n_grid
        self.grid1D = np.zeros((n_grid,2))
        Cta_ary = np.arange(np.pi/n_grid,2.*np.pi,2.*np.pi/n_grid)
        self.grid1D[:,0] = radius*np.cos(Cta_ary)
        self.grid1D[:,1] = radius*np.sin(Cta_ary)

    #generating the linear interpolation function based on rbf
    def gene_rbf_func(self,z_ary,type = 'multiquadric',epsilon=2):
        self.function = Rbf(self.x_ary,self.y_ary,z_ary, function=type,epsilon = epsilon)

    #generating the linear interpolation function based on linear interpolation
    def gene_linear_func(self,z_ary,type = 'linear'):
        self.function = itp.interp2d(self.x_ary,self.y_ary,z_ary,kind=type)

    #generating boundary value
    def compute_boundary(self):
        self.bound_value = np.zeros((self.n_grid1D))
        for i in range(self.n_grid1D):
            self.bound_value[i] = self.function(self.grid1D[i][0],self.grid1D[i][1])
        return self.bound_value

    #for 2-D plotting use
    def plot_function(self,radius,n_mesh,min,max,title="linear interpolation"):
        xx = yy = np.arange(-radius,radius,2.*radius/n_mesh)
        X,Y = np.meshgrid(xx,yy)
        Z = self.function(X,Y)
        exterior = np.sqrt((X**2 +Y**2)) > radius
        Z[exterior] = np.nan
        levs = np.arange(min,max,(max-min)/20.)
        cs = plt.contourf(X, Y, Z, levs, cmap="coolwarm", extend="both", origin="lower")
        plt.title(title,family = 'serif',size = 17, y=1.04)
        plt.colorbar(cs)
        plt.axes().set_aspect('equal')
        plt.show()


