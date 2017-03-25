#!/usr/bin/env python

import numpy as np
from matplotlib import pylab
from fipy import *
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate


##Create mesh for the Polar system (lnR,Theta)

nLnR = 100
nTheta = 50
dLnR = 0.02
dTheta = 2*np.pi/nTheta
L = dLnR * nLnR
mesh = PeriodicGrid2DTopBottom(dx=dLnR, dy=dTheta, nx=nLnR, ny=nTheta)

##Coordinates
LnR= mesh.cellCenters[0]
LnRface= mesh.faceCenters[0]
Theta= mesh.cellCenters[1]
Thetaface= mesh.faceCenters[1]

##Create mesh for the Cartesian system (X,Y)

nx=100
ny=nx
dx=1.0
dy=dx
L1=nx*dx
mesh1 = Grid2D(dx=dx,dy=dy,nx=nx,ny=ny)

##Coordinates
X,Y = mesh1.cellCenters


#Constants

E_f = 1.80

C0V = 3.62E-10

C0Si = 5E-3

mu0= 0.00

Sigma = -15.60

alphaV = 18.1

alphaSi = -0.251

KB = 8.617E-5

Temp = 960

D0VV = 0.152

D1VV = 12.9/3.0

DLVV = -7.42

D0SiV = 0.157

D1SiV = 13.3/3.0

DLSiV = -22.4

D0SiSi = 1.29

D1SiSi = 109/3.0

DLSiSi = -50.8

b = 2.5

R0 = 50/np.exp(L)

rate = 0.25 ##vacancy-dislocation anihilation rate

Coeff_Saddle = 0.07  ##Linear Coefficient between volumetric strain and saddle point energy

E_Saddle_0 = 2.8 ##Saddle point energy without volumetric strain

Sdl = 1.0 ##for calculating the saddle point energy

nu_0 = 5E4  ##attempt frequency

R2_Nei = 1.5

##Define Volumetric Strains and other variables
Radi = R0*np.exp(LnR)
Radiface = R0*np.exp(LnRface)

RSquare = Radi**2
RSquareface = Radiface**2
RSquareSq = (X-L1/2.0)**2+(Y-L1/2.0)**2

COSY = (X-L1/2.0)/np.sqrt(RSquareSq)
SINY = (Y-L1/2.0)/np.sqrt(RSquareSq)
COS2Y = ((X-L1/2.0)**2-(Y-L1/2.0)**2)/(X-L1/2.0)**2+(Y-L1/2.0)**2
SIN2Y = 2.0*COSY*SINY
COS3Y = (4*COSY**3-3*COSY)

epsilonS = -b*(Y-L1/2.0)/(4.0*np.pi*RSquareSq)
epsilonP = -b*np.sin(Theta)/(4.0*np.pi*Radi)
epsilonface = -b*np.sin(Thetaface)/(4.0*np.pi*Radiface)


def PA(x):
    return x%(2*np.pi)

##Initialization of the field Variables
def Initialization(MuV,Cbg):
    MuV = CellVariable(name="Vacancy Chemical Potential",mesh=mesh,value=0.001,hasOld=1)
    MuV.setValue(alphaV*epsilonP+KB*Temp*np.log(Cbg/C0V))
    CV = CellVariable(name = "Concentration",mesh=mesh,value = 0.0)
    CV.setValue(C0V*np.exp((MuV-alphaV*epsilonP)/KB/Temp))
    return CV

##Plot Vacancy Concentration
def V_Viewer(CVxy,Cbg):
    return Matplotlib2DGridContourViewer(vars = CVxy*1E6,
                              datamin = 0.00,
                              datamax = Cbg*1E6)

##Linear Interpolation used in the Inverse transformation
def LinearInter(MuV,MuVxy,CBG):
    Xcenter=0.5*np.log((X-L1/2)**2+(Y-L1/2)**2)-np.log(R0)
    Ycenter=np.arctan2(Y-L1/2,X-L1/2)
    Nx=np.floor((Xcenter-dLnR/2.0)/dLnR)
    Xleft=np.floor((Xcenter-dLnR/2.0)/dLnR)*dLnR+0.5*dLnR
    Xright=Xleft+dLnR
    Ny=np.floor((Ycenter-dTheta/2.0)/dTheta)
    Ybottom=np.floor((Ycenter-dTheta/2.0)/dTheta)*dTheta+0.5*dTheta
    Ytop=Ybottom+dTheta
    UX=(Xright-Xcenter)/dLnR
    UY=(Ytop-Ycenter)/dTheta
    MuVxy.setValue((UX*UY*MuV((Xleft,PA(Ybottom)))+(1-UX)*UY*MuV((Xright,PA(Ybottom)))+UX*(1-UY)*MuV((Xleft,PA(Ytop)))+(1-UX)*(1-UY)*MuV((Xright,PA(Ytop)))))
    MuVxy.setValue(alphaV*epsilonS+KB*Temp*np.log(CBG/C0V), where=((X-L1/2)**2+(Y-L1/2)**2>2500))
    MuVxy.setValue(mu0, where=((X-L1/2)**2+(Y-L1/2)**2<R0**2))
    return MuVxy

##Calculate the Concentration from the Chemical Potential
def CP_Concentration(MuVxy,CVxy):
    CVxy.setValue(C0V*np.exp((MuVxy-alphaV*epsilonS)/KB/Temp))
    return CVxy

##Calculation of the Adaptive time step
def AdaptiveStep(DV,alphaT):
    return alphaT/max(DV)

##Calculate the total Inner Flux
def Vflux(MuV,alp):

    DVVP = (D0VV + D1VV*epsilonP)

    CV = CellVariable(name = "Concentration",mesh=mesh,value = 0.0)
    
    CV.setValue(C0V*np.exp((MuV-alp*epsilonP)/KB/Temp))

    InFlux = CellVariable(name = "1DFunction", mesh = mesh,value=0.05)

    InFlux.setValue(CV*DVVP*MuV.grad[0])

    ##Need to Create a 1-D mesh to store a 1-D function

    nCir = 50
    dCir = 2*np.pi/nCir
    mesh1D = Grid1D(dx=dCir,nx=nCir)
    Cir, = mesh1D.cellCenters
    
    V = CellVariable(name="Vector components",mesh=mesh1D,value=0.)
    
    V.setValue(InFlux((Cir*0.0,Cir*1.0)))

    Sum = 0.0
    
    for i in range(0,nCir):
        Sum += V.getValue()[i]
    return(Sum)

##Output the data of the concentration value with respect to R
##For convenience, we plot G(r,theta), which can be directly obtained via the analytical method
def VC_RPlot(MuV,theta):
    GV = CellVariable(name = "GV",mesh=mesh,value = 0.0)
    GV.setValue(np.exp(MuV/KB/Temp))
    ##Need to Create a 1-D mesh to store a 1-D function
    nRad = 50
    dRad = 0.04
    mesh1D = Grid1D(dx=dy1,nx=ny1)
    Rad, = mesh1D.cellCenters
    U = CellVariable(name="Concentration components",mesh=mesh1D,value=0.)
    U.setValue(ConV((Rad*1.0,Rad*0.0+theta)))
    ##print G(r,theta) versus r
    for i in range(0,nRad):
        Rreal = R0*np.exp((i+0.5)*dRad)
        print Rreal,U.getValue()[i]
    return GV

def Ave_CV(MuV):

    CV = CellVariable(name = "Concentration",mesh=mesh,value = 0.0)
    
    CV.setValue(C0V*np.exp((MuV-alp*epsilonP)/KB/Temp))

    ##Need to Create a 1-D mesh to store a 1-D function

    nCir = 50
    dCir = 2*np.pi/nCir
    mesh1D = Grid1D(dx=dCir,nx=nCir)
    Cir, = mesh1D.cellCenters
    
    Ccore = CellVariable(name="Vector components",mesh=mesh1D,value=0.)
    
    Ccore.setValue(CV((Cir*0.0,Cir*1.0)))

    Sum = 0.0
    
    for i in range(0,nCir):
        Sum += V.getValue()[i]
    return(Sum/nCir)

'''
*****************
Functions below are used for coupling between Discretized region and Continuum region
'''

##Compute the saddle point energies, which is assumed to be linearly related to volumetric strain
def MigEner(E1,E2):          
    return (max(E1,E2)-E1+Sdl)

##Given x and y, calculate the volumetric strain
def VolStrain(x,y):
    R2 = x**2 + y**2
    eps = -y*b/4/np.pi/R2
    return(eps)
'''
Generate the source term for the discretized equations, based on the output from the last iteration step:
Cval: the concentration field in the continuum region
CMatrix is the concentraion at the inner sites
InternalSite is the sites in the core region
ExternalSite is the sites outside the core region (next to the boundary)
Flux is the total inner flux computed from the last iteration step
'''
def ExSourceTerm(Cval, CMatrix, InternalSite, ExternalSite):
    Nin = np.shape(InternalSite)[0]
    Nout = np.shape(ExternalSite)[0]
    Prob = [0 for i in xrange(Nin)]
    for i in range(Nout):
        Xo = ExternalSite.item((i,0))
        Yo = ExternalSite.item((i,1))
        Co = Cval((Xo+L1/2,Yo+L1/2))
        for j in range(np.shape(InternalSite)[0]):
            Xi = InternalSite.item((j,0))
            Yi = InternalSite.item((j,1))
            dr2 = (Xo-Xi)**2 + (Yo-Yi)**2
            if(dr2<R2_Nei*b**2):
                Ci = CMatrix.item((j))
                epo = VolStrain(Xo,Yo)
                epi = VolStrain(Xi,Yi)
                Eso = alphaV*epo + E_f
                Esi = alphaV*epi + E_f
                nu_io = nu_0*np.exp(-MigEner(Esi,Eso)/KB/Temp)
                nu_oi = nu_0*np.exp(-MigEner(Eso,Esi)/KB/Temp)
                Prob[j] += Co*(1-Ci)*nu_oi - Ci*(1-Co)*nu_io
    SMatrix = np.matrix(Prob)                 ##The source term is in the format of np.matrix
    return(SMatrix)


'''
Create Dirichlet B.C. for the Continuum region:
Blst is the indexes of the sites that are very close to the boundary
Anglst is the angular value corresponding to the sites in Blst
Cut[i] is the value at ith point on the boundary
'''
##Calculate values for separate points in the Circle
def CutBoundary(Cval,CMatrix,InternalSite,Blst,Anglst):
    NCta = len(Blst)
    Cut=[0 for i in xrange(NCta)]
    for i in range(NCta):
        Xi = InternalSite.item((Blst[i],0))
        Yi = InternalSite.item((Blst[i],1))
        Xb = R0*np.cos(Anglst[i])
        Yb = R0*np.sin(Anglst[i])
        Xo = 2*Xb - Xi
        Yo = 2*Yb - Yi
        Cut[i] = 0.5*(CMatrix.item((Blst[i])) + Cval((Xo+L1/2,Yo+L1/2)))
    return(Cut)

##Compute exact values for the inner B.C.
def InDirichlet(Anglst,Cut):
    nCta = 50
    dCta = 2*np.pi/nCta
    mesh1D = Grid1D(dx=dCta,nx=nCta)
    Cta, = mesh1D.cellCenters
    N = len(Anglst)-1
    
    Inner = CellVariable(name="Inner B.C.",mesh=mesh1D,value=0.)

# When Cta<Anglst[0]
    b = Anglst[0] + 2.0*np.pi
    a = Anglst[N-1]
    lmda = (b-Cta-2.0*np.pi)/(b-a)
    Inner.setValue(Cut[N-1]*lmda + Cut[N]*(1-lmda), where = (Cta >= 0)&(Cta < Anglst[0]))

# When Cta>Anglst[N]
    b = Anglst[0] + 2.0*np.pi
    a = Anglst[N-1]
    lmda = (b-Cta)/(b-a)
    Inner.setValue(Cut[N-1]*lmda + Cut[N]*(1-lmda), where = (Cta >= Anglst[N])&(Cta < 2.0*np.pi))

# When Anglst[0]<Cta<Anglst[N]
    for i in range(N-1):
        lmda = (Anglst[i+1]-Cta)/(Anglst[i+1]-Anglst[i])
        Inner.setValue(Cut[i]*lmda+Cut[i+1]*(1-lmda), where = (Cta >= Anglst[i])&(Cta < Anglst[i+1]))

    return(Inner) ##Inner is the InnerBC for the Continuum equation build by linear interpolation


###Using scipy.interpolate functions but Error is larger than the previous one
def SpyIntBC(Inxy,Exxy,C_dis,C_con):
    N1 = np.shape(Inxy)[0]
    N2 = np.shape(Exxy)[0]
    N = N2 + N2
    XX = [0 for i in xrange(N)]
    YY = [0 for i in xrange(N)]
    ZZ = [0 for i in xrange(N)]
    for i in range(N1):
        XX[i]=Inxy.item((i,0))
        YY[i]=Inxy.item((i,1))
        ZZ[i]=C_dis.item(i)
    for i in range(N1,N):
        XX[i]=Exxy.item((i-N1,0))
        YY[i]=Exxy.item((i-N1,1))
        ZZ[i]=C_con((XX[i]+L1/2,YY[i]+L1/2))
    f_xy = interpolate.interp2d(XX,YY,ZZ,kind='linear')
    
    ##Create mesh for InnerBC
    nCta = 50
    dCta = 2*np.pi/nCta
    mesh1D = Grid1D(dx=dCta,nx=nCta)
    Cta, = mesh1D.cellCenters

    Inner = CellVariable(name="Inner B.C.",mesh=mesh1D,value=0.)

    BValue = [0 for i in xrange(len(Cta))]
    for i in range(len(Cta)):
        Bvalue[i] = f_xy(R0*np.cos(Cta[i]),R0*np.sin(Cta[i]))[0]
    Inner.setValue(BValue)
    return(Inner)


        
    
    
    



