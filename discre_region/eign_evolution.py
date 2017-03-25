__author__ = 'zeboli1'

import numpy as np
import scipy
from scipy import linalg as LA
import Parameter as PR

nu = 1E-3
barrier = 1.073
kT = PR.KB*PR.Temp
d_t0 = 1E-3#Initial time_step


fpc = open('Inputs/Inner_sites.txt','r')
Cd_in = [] #coordinates of inner sites
for line in fpc:
    Cd_in.append(map(float,line.split(' ')))
fpc.close()

Nst = len(Cd_in) #Number of inner sites

Nbor = [] #Neighbor information
fpn = open('Inputs/neighbor.txt','r')
for line in fpn:
    Nbor.append(map(int,line.split(' ')))
fpn.close()

fpe = open('Inputs/site_energy.txt','r')
E_V = map(float,fpe.read().split(' '))

##Create Jump array
JumpAry = np.zeros((Nst,Nst))
for i in range(Nst):
    for j in range(Nst):
        if(Nbor[i][j] == 1):
            E_ini = E_V[i]
            E_fin = E_V[j]
            E_mid = 0.5*(E_ini + E_fin) + barrier
            E_bar = max(E_ini,E_mid,E_fin) - E_ini
            JumpAry[j][i] = nu*np.exp(-E_bar/kT)
for i in range(Nst):
    JumpAry[i][i] = 0.0
    for j in range(Nst):
        if(j != i):
            JumpAry[i][i] += -JumpAry[j][i]





def Mdfy(J,C):
    V = np.zeros((Nst,Nst))
    for i in range(Nst):
        V[i][i] = 0.0
        for j in range(Nst):
            if(i != j):
                V[j][i] = J[j][i]*(1.0 - C[j])
                V[i][i] += -V[j][i]
    return(V)

def get_eigen_info(M):
    Ary = np.array(M)
    wr,vl,vr = LA.eig(Ary,left=True, right=True)
    idx = wr.argsort()[::-1]
    val = wr[idx]             #sorted eigenvalues
    vec_left = vl[:,idx]
    vec_right = vr[:,idx]           #right eigenvectors
    #Normalized left eigenvectors
    Gamma = np.dot(vec_left.transpose(),vec_right)
    D = []
    for i in range(Nst):
        D.append(1./Gamma[i][i])
    O = np.diag(D)
    vec_left = np.dot(O,vec_left.transpose()) #so that vec_right*vec_left^T = I
    return val,vec_right,vec_left

def make_symmetry(Xary,n_ary):
    i = 0
    while(i<n_ary-1):
        s = Xary[i] + Xary[i+1]
        Xary[i] = 0.5*s
        Xary[i+1] = 0.5*s
        i += 2


def Onestep(x,dt):
    MdJump = Mdfy(JumpAry,x)
    eigen = get_eigen_info(MdJump)
    initial_values = np.array(eigen[0])
    eig_vec_right = eigen[1]
    eig_vec_left = eigen[2]

    Coeff = np.dot(eig_vec_left,x)
    for i in range(Nst):
        Coeff[i] = Coeff[i]*np.exp(initial_values[i]*dt)
    x1 = np.dot(eig_vec_right,Coeff)

    x_midpt = 0.5*(x1+x)
    make_symmetry(x_midpt,Nst)

    MdJump = Mdfy(JumpAry,x_midpt)
    eigen = get_eigen_info(MdJump)
    media_values = np.array(eigen[0])
    eig_vec_right = eigen[1]
    eig_vec_left = eigen[2]

    Coeff = np.dot(eig_vec_left,x)
    for i in range(Nst):
        Coeff[i] = Coeff[i]*np.exp(media_values[i]*dt)
    x1 = np.dot(eig_vec_right,Coeff)
    make_symmetry(x1,Nst)

    MdJump = Mdfy(JumpAry,x1)
    eigen = get_eigen_info(MdJump)
    final_values = np.array(eigen[0])

    d_values = final_values - initial_values

    adp = dt*np.dot(final_values,final_values)/np.dot(d_values,d_values)

    return x1,adp
fp = open('output.txt','w')

Nstp = 5000000
d_t = d_t0
Nadp = 100
Tadp = [d_t for i in xrange(Nadp)]
Xlst = [5.E-2 for i in xrange(Nst)]
X = np.array(Xlst)
Time = 0.0
n = 0
while(Time < 1E10):
    rst = Onestep(X,d_t)
    X = rst[0]
    Tadp[n%Nadp] = 5.2E-4*rst[1]
    if((n+1)%Nadp == 0):
        Time += Nadp*d_t
        d_t = min(1.2*d_t,sum(Tadp)/Nadp)
        print n,Time,d_t,sum(X),X
        fp.write(str(Time) + ' ' + str(d_t) + '\n')
    n += 1


