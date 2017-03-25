__author__ = 'zeboli1'

import fipy
from fipy import*
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from fipy import Viewer
import time

L_total = 50.
nx = 100
dx = 0.5
mesh_all = Grid1D(nx = nx, dx = dx)
x_al = mesh_all.cellCenters[0]


L_bf = 6.
L_half = 0.5*(L_total + L_bf)
nx_hf = 100
dx_hf = L_half/nx_hf
mesh_half = Grid1D(nx = nx_hf, dx = dx_hf)
x_hf = mesh_half.cellCenters[0]



D = 1.

valueLeft = 0.
valueRight = 50.

d_t = 10. * dx**2 / (2 * D)
N_st = 5

def evolution_left(phi_left,rightBC,Nstep):
    phi_left.constrain(valueLeft,mesh_half.facesLeft)
    phi_left.constrain(rightBC,mesh_half.facesRight)
    eq_left = TransientTerm(coeff=1.,var = phi_left) == DiffusionTerm(coeff=D,var=phi_left)
    for i in range(Nstep):
        eq_left.solve(dt=d_t)


def evolution_right(phi_right,leftBC,Nstep):
    phi_right.constrain(leftBC,mesh_half.facesLeft)
    phi_right.constrain(valueRight,mesh_half.facesRight)
    eq_right = TransientTerm(coeff=1.,var = phi_right) == DiffusionTerm(coeff=D,var=phi_right)
    for i in range(Nstep):
        eq_right.solve(dt=d_t)

def evolution_all(phi_all,Nstep):
    phi_all.constrain(valueLeft,mesh_all.facesLeft)
    phi_all.constrain(valueRight,mesh_all.facesRight)
    eq_all = TransientTerm(coeff=1.,var = phi_all) == DiffusionTerm(coeff=D,var=phi_all)
    for i in range(Nstep):
        eq_all.solve(dt=d_t)

def combine(phi_whole,phi_left,phi_all):
    phi_whole.setValue(phi_left((x_al,)), where = x_al < (L_half - L_bf))
    phi_whole.setValue(phi_right((x_al-(L_half - L_bf),)), where = x_al >= L_half )
    phi_whole.setValue(0.5*phi_left((x_al,))+0.5*phi_right((x_al-(L_half - L_bf),)),
                       where = (x_al>=(L_half - L_bf))&(x_al< L_half))


def Error(f1,f2):
    sum = 0
    er = 0
    for i in range(nx):
        sum += f1.getValue()[i]**2
        er += (f1.getValue()[i]-f2.getValue()[i])**2
    s = np.sqrt(sum)
    e = np.sqrt(er)
    return(e/s)



Niter = 100

if __name__ == '__main__':
    phi_left = CellVariable(name = "left_half",mesh = mesh_half,value=0.)
    phi_right = CellVariable(name = "right_half",mesh = mesh_half,value=0.)

    phi_all = CellVariable(name="One_variable",mesh=mesh_all,value=0.0)
    phi_whole = CellVariable(name="As_a_whole",mesh=mesh_all,value=0.0)

    print "Ilove HyoJung"
    viewer = Matplotlib1DViewer(vars=(phi_all,phi_whole), datamin=-5., datamax=55.)
    #viewer = Matplotlib1DViewer(vars=(phi_left,phi_right), datamin=-27., datamax=27.)
    fp = open('error_analysis/Buff=6.txt','w')

    Time = 0.0




    for n in range(Niter):
        a = time.time()
        RightValue = phi_right((L_bf,))
        LeftValue = phi_left((L_half-L_bf,))
        print "I Love HyoJung Kim forever!!"
        evolution_right(phi_right,LeftValue,N_st)
        evolution_left(phi_left,RightValue,N_st)
        combine(phi_whole,phi_left,phi_right)
        evolution_all(phi_all,N_st)
        Time += d_t*N_st
        fp.write("%.12f"%Time + ' ' + "%.12f"%Error(phi_all,phi_whole))
        if(n!=(Niter-1)):
            fp.write('\n')
        #print Error(phi_all,phi_whole)
        viewer.plot()
        b = time.time()
        print n,b-a,Error(phi_all,phi_whole)
    viewer._promptForOpinion()

