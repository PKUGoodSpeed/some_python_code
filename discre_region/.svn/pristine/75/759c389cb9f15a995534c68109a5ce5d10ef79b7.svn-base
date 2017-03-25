__author__ = 'zeboli1'

import fipy
from fipy import*
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from fipy import Viewer
import time


r_0 = 2.
r_1 = 12.
d_bf = 10.
r_2 = 60.
C = 5.




L_total = r_2 - r_0
nx = 400
dx = L_total/nx
mesh_all = Grid1D(nx = nx, dx = dx)
x_al = mesh_all.cellCenters[0]


L_left = r_1 + d_bf - r_0
nx_left = 400
dx_left = L_left/nx_left
mesh_left = Grid1D(nx = nx_left, dx = dx_left)
x_lt = mesh_left.cellCenters[0]

L_right = r_2 - r_1
nx_right = 400
dx_right = L_right/nx_right
mesh_right = Grid1D(nx = nx_right, dx = dx_right)
x_rt = mesh_right.cellCenters[0]

L_2D = 2*r_2
nx_2D = 480
ny_2D = nx_2D
dx_2D = L_2D/nx_2D
dy_2D = dx_2D
mesh_2D = Grid2D(dx=dx_2D, dy=dy_2D, nx=nx_2D, ny=ny_2D)
##Coordinates
X_2D,Y_2D = mesh_2D.cellCenters


D_all = x_al + r_0
D_left = x_lt + r_0
D_right = x_rt + r_1

valueLeft = C*np.log(r_0)
valueRight = C*np.log(r_2)



d_t = 192. * dx**2 / (2 * r_1)
N_st = 2

def evolution_left(phi_left,rightBC,Nstep):
    phi_left.constrain(valueLeft,mesh_left.facesLeft)
    phi_left.constrain(rightBC,mesh_left.facesRight)
    eq_left = TransientTerm(coeff=1.,var = phi_left) == DiffusionTerm(coeff=D_left,var=phi_left)
    for i in range(Nstep):
        eq_left.solve(dt=d_t)


def evolution_right(phi_right,leftBC,Nstep):
    phi_right.constrain(leftBC,mesh_right.facesLeft)
    phi_right.constrain(valueRight,mesh_right.facesRight)
    eq_right = TransientTerm(coeff=1.,var = phi_right) == DiffusionTerm(coeff=D_right,var=phi_right)
    for i in range(Nstep):
        eq_right.solve(dt=d_t)

def evolution_all(phi_all,Nstep):
    phi_all.constrain(valueLeft,mesh_all.facesLeft)
    phi_all.constrain(valueRight,mesh_all.facesRight)
    eq_all = TransientTerm(coeff=1.,var = phi_all) == DiffusionTerm(coeff=D_all,var=phi_all)
    for i in range(Nstep):
        eq_all.solve(dt=d_t)

def combine(phi_whole,phi_left,phi_right):
    phi_whole.setValue(phi_left((x_al,)), where = x_al < (L_left - d_bf))
    phi_whole.setValue(phi_right((x_al-(L_left - d_bf),)), where = x_al >= L_left )
    phi_whole.setValue(0.5*phi_left((x_al,))+0.5*phi_right((x_al-(L_left - d_bf),)),
                       where = (x_al>=(L_left - d_bf))&(x_al< L_left))

def trans_2D(phi_2D,phi_whole):
    r = np.sqrt((X_2D-r_2)**2 + (Y_2D-r_2)**2)
    phi_2D.setValue(0.0, where = r<r_0)
    phi_2D.setValue(0.0, where = r>r_0)
    phi_2D.setValue(phi_whole((r-r_0,)), where = (r>=r_0) & (r<=r_2))


def Error(f1,f2):
    sum = 0
    er = 0
    for i in range(nx):
        sum += f1.getValue()[i]**2
        er += (f1.getValue()[i]-f2.getValue()[i])**2
    s = np.sqrt(sum)
    e = np.sqrt(er)
    return(e/s)



Niter = 60

if __name__ == '__main__':
    phi_left = CellVariable(name = "left_half",mesh = mesh_left,value=valueLeft)
    phi_right = CellVariable(name = "right_half",mesh = mesh_right,value=valueRight)

    phi_all = CellVariable(name="One_variable",mesh=mesh_all,value=valueLeft)
    phi_whole = CellVariable(name="As_a_whole",mesh=mesh_all,value=valueRight)

    phi_2D = CellVariable(name = "2D_function",mesh = mesh_2D,value = 0.)

    phi_left.setValue(valueLeft)
    phi_right.setValue(valueLeft)
    combine(phi_whole,phi_left,phi_right)
    #phi_all.setValue(phi_whole)
    trans_2D(phi_2D,phi_whole)
    print "Ilove HyoJung"
    #viewer = Matplotlib1DViewer(vars=(phi_all,phi_whole), datamin=valueLeft-0.5, datamax=valueRight+0.5)
    viewer = Matplotlib2DGridContourViewer(vars = phi_2D, datamin = valueLeft-0.2, datamax = valueRight+2.0,width = 1, height = 4.0)
    for i in range(3000):
        print i,Error(phi_all,phi_whole)


    for n in range(Niter):
        a = time.time()
        RightValue = phi_right((d_bf,))
        LeftValue = phi_left((L_left-d_bf,))
        print "I Love HyoJung Kim forever!!"
        evolution_right(phi_right,LeftValue,N_st)
        evolution_left(phi_left,RightValue,N_st)
        combine(phi_whole,phi_left,phi_right)
        #evolution_all(phi_all,N_st)
        trans_2D(phi_2D,phi_whole)
        #print Error(phi_all,phi_whole)
        viewer.plot()
        b = time.time()
        print n,(b-a)
    viewer._promptForOpinion()

