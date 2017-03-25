__author__ = 'zeboli1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, ticker, cm
from fipy import*

class vac_conc_2Dcontour:
    #for vacancy concentraion contour plot use
    def __init__(self,length,n_grid):
        self.length = length
        self.delta = length/n_grid

    #making contour plot
    def get_color_map(self,upper,lower,style = "coolwarm"):
        self.cmap = plt.cm.get_cmap(style)
        self.cmap.set_under(self.cmap(0))
        self.cmap.set_over(self.cmap(400))
        self.up = upper
        self.low = lower

    '''
    making a single contour plot
    '''
    def making_contour(self,fipy_variable,inner_radi,outer_radi,title = " ", color_bar = False):
        origin = 'lower'
        x = y = np.arange(-self.length/2.,self.length/2.,self.delta)
        X,Y = np.meshgrid(x,y)
        Z = fipy_variable((X+self.length/2.,Y+self.length/2.))

        interior = np.sqrt((X**2 +Y**2)) < inner_radi
        exterior = np.sqrt((X**2 +Y**2)) > outer_radi
        Z[interior] = np.nan
        Z[exterior] = np.nan

        levs = np.arange(self.low,self.up,(self.up-self.low)/20.)

        cs = plt.contourf(X, Y, Z, levs, cmap=self.cmap, extend="both", origin=origin)

        plt.title(title,family = 'serif',size = 32, y=1.04)
        labels = ["-40","-20","0","20","40"]
        plt.xticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length],labels,rotation=0,family = 'serif',size = 22)
        plt.yticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length],labels,rotation=0,family = 'serif',size = 22)
        if(color_bar == True):
            plt.colorbar(cs, extend="both", spacing="proportional",
                orientation="horizontal", shrink=1.0 )
        plt.axes().set_aspect('equal')
        plt.show()

    '''
    making multiple plots in a horizontal orientation 2X3
    '''
    def making_multi_hori(self,fipy_vari_list,inner_radi,outer_radi,label_list,width,height,color_bar = True):
        origin = 'lower'
        x = y = np.arange(-self.length/2.,self.length/2.,self.delta)
        X,Y = np.meshgrid(x,y)
        interior = np.sqrt((X**2 +Y**2)) < inner_radi
        exterior = np.sqrt((X**2 +Y**2)) > outer_radi


        Z = []
        for i in range(6):
            Z.append(fipy_vari_list[i]((X+self.length/2.,Y+self.length/2.)))
            Z[i][interior] = np.nan
            Z[i][exterior] = np.nan

        levs = np.arange(self.low,self.up,(self.up-self.low)/20.)

        fig, axs = plt.subplots(2, 3,figsize=(width, height))

        for i in range(6):
            cs = axs[i/3][i%3].contourf(X, Y, Z[i], levs, cmap=self.cmap, extend="both", origin=origin)
            #axs[i/3][i%3].set_title("$\epsilon_V$",family = 'serif',size = 32, y=1.04)
            axs[i/3][i%3].set_xticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length])
            axs[i/3][i%3].set_xticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
            axs[i/3][i%3].set_xlabel(label_list[i],family = 'serif', size = 25, y = 1.4)
            axs[i/3][i%3].set_yticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length])
            axs[i/3][i%3].set_yticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
            axs[i/3][i%3].set_aspect('equal')
        if(color_bar == True):
            vmin,vmax = cs.get_clim()
            #-- Defining a normalised scale
            cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            #-- Creating a new axes at the right side
            ax_add = fig.add_axes([0.9, 0.1, 0.017, 0.8])
            #-- Plotting the colormap in the created axes
            cb = mpl.colorbar.ColorbarBase(ax_add,cmap=self.cmap, norm=cNorm , extend = "both")
            ax_add.get_yaxis().set_ticks([])
            #for j, lab in enumerate(['$-0.03$','$-0.01$','$0.01$','$0.03$']):
            #    ax_add.text(2.5, j/3.0*19./20. + 1./40, lab, ha='center', va='center', family = 'serif',size = 22)

            fig.subplots_adjust(left=0.05,right=0.85)
        plt.show()


    '''
    making multiple plots in a vertical orientation 3X2
    '''
    def making_multi_vert(self,fipy_vari_list,inner_radi,outer_radi,label_list,width,height,color_bar = True):
        origin = 'lower'
        x = y = np.arange(-self.length/2.,self.length/2.,self.delta)
        X,Y = np.meshgrid(x,y)
        interior = np.sqrt((X**2 +Y**2)) < inner_radi
        exterior = np.sqrt((X**2 +Y**2)) > outer_radi


        Z = []
        for i in range(6):
            Z.append(fipy_vari_list[i]((X+self.length/2.,Y+self.length/2.)))
            Z[i][interior] = np.nan
            Z[i][exterior] = np.nan

        levs = np.arange(self.low,self.up,(self.up-self.low)/20.)

        fig, axs = plt.subplots(3, 2,figsize=(width, height))

        for i in range(6):
            cs = axs[i/2][i%2].contourf(X, Y, Z[i], levs, cmap=self.cmap, extend="both", origin=origin)
            #axs[i/3][i%3].set_title("$\epsilon_V$",family = 'serif',size = 32, y=1.04)
            axs[i/2][i%2].set_xticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length])
            axs[i/2][i%2].set_xticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
            axs[i/2][i%2].set_xlabel(label_list[i],family = 'serif', size = 25, y = 1.4)
            axs[i/2][i%2].set_yticks([-4./10.*self.length,-2./10.*self.length,0,2./10.*self.length,4./10.*self.length])
            axs[i/2][i%2].set_yticklabels(["-40","-20","0","20","40"],rotation=0,family = 'serif',size = 22 )
            axs[i/2][i%2].set_aspect('equal')
        if(color_bar == True):
            vmin,vmax = cs.get_clim()
            #-- Defining a normalised scale
            cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            #-- Creating a new axes at the right side
            ax_add = fig.add_axes([0.9, 0.1, 0.017, 0.8])
            #-- Plotting the colormap in the created axes
            cb = mpl.colorbar.ColorbarBase(ax_add,cmap=self.cmap, norm=cNorm , extend = "both")
            ax_add.get_yaxis().set_ticks([])
            #for j, lab in enumerate([' ',' ',' ',' ']):
            #    ax_add.text(2.5, j/3.0*19./20. + 1./40, lab, ha='center', va='center', family = 'serif',size = 22)

            fig.subplots_adjust(left=0.05,right=0.85)
        plt.show()
