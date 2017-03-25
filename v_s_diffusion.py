__author__ = 'zeboli1'

import numpy as np
import fipy
from fipy import *
from fipy import Viewer
from matplotlib import pylab
import matplotlib.pyplot as plt
import Parameter as PR
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, ticker, cm
import plotting_functions as PF

nLnR = 50
nCta = 50
dLnR = 0.04
dCta = 2*np.pi/nCta
PL = nLnR*dLnR
##Create mesh in Polar Coordinate system (LnR,Cta)
Pmesh = PeriodicGrid2DTopBottom(dx=dLnR, dy=dCta, nx=nLnR, ny=nCta)
Psudomesh = Grid2D(dx=dLnR, dy=dCta, nx=nLnR, ny=nCta)
##Coordinates
LnR = Pmesh.cellCenters[0]
LnRf = Pmesh.faceCenters[0]
Cta = Pmesh.cellCenters[1]
Ctaf = Pmesh.faceCenters[1]

##Create mesh in the Cartesian Coordinate system (X,Y)
nx = 2000
ny = nx
dx = 100.0/nx
dy = dx
CL = nx*dx
Cmesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
##Coordinates
X,Y = Cmesh.cellCenters



Mu_0 = PR.mu0
Sigma = PR.Sigma
alpha_V = PR.alphaV
alpha_S = PR.alphaSi
KB = PR.KB
Temp = PR.Temp
KT = KB*Temp
##Vacancy term
D0_VV = PR.D0VV
Dv_VV = PR.D1VV
Dt_VV = PR.DLVV
##Solute-vacancy coupling
D0_SV = PR.D0SiV
Dv_SV = PR.D1SiV
Dt_SV = PR.DLSiV
##Solute-term
D0_SS = PR.D0SiSi
Dv_SS = PR.D1SiSi
Dt_SS = PR.DLSiSi

#region properties
b = PR.b
R_out = 50
R_in = R_out/np.exp(PL)

##some additional energy
E_f = PR.E_f
E_min = -1.081

##define radius
r = R_in*np.exp(LnR)
rf = R_in*np.exp(LnRf)

C_S_far = PR.C0Si
C_Eq = np.exp(-E_f/KT)

#define strains
###volumetric
ep_V = -b*np.sin(Cta)/(4.0*np.pi*r)
ep_Vf = -b*np.sin(Ctaf)/(4.0*np.pi*rf)
###tetragonal1
ep_bb = -b*(4.0+3.0*np.cos(2.0*Cta))*np.sin(Cta)/(8.0*np.pi*r)
###tetragonal2
ep_bn = 3.0*b*(np.cos(Cta)+np.cos(3*Cta))/(16.0*np.pi*r)

RSq_C = (X-CL/2.0)**2+(Y-CL/2.0)**2
ep_V_C = -b*(Y-CL/2.0)/(4.0*np.pi*RSq_C)

#define Inverse Transformation

def PA(x):
    return x%(2*np.pi)

def CEq_FD(E):
    return(1/(1+np.exp(E/KT)))




##Intermediated matrices
I_off = numerix.array(((np.cos(Cta)**2,-np.sin(Cta)*np.cos(Cta)),(np.sin(Cta)*np.cos(Cta),np.cos(Cta)**2)))

I_10 = numerix.array(((np.sin(2*Cta),np.cos(2*Cta)),(np.cos(2*Cta),-np.sin(2*Cta))))

#define convection velocities and diffusion term coefficients
#intermediate terms
Diff_VV = (D0_VV+Dv_VV*ep_V)*numerix.array(((1,0),(0,1)))+ (1.0/6.0*Dt_VV*ep_bb)*I_off + (2.0/3.0*Dt_VV*ep_bn)*I_10
Diff_SV = (D0_SV+Dv_SV*ep_V)*numerix.array(((1,0),(0,1)))+ (1.0/6.0*Dt_SV*ep_bb)*I_off + (2.0/3.0*Dt_SV*ep_bn)*I_10
Diff_SS = (D0_SS+Dv_SS*ep_V)*numerix.array(((1,0),(0,1)))+ (1.0/6.0*Dt_SS*ep_bb)*I_off + (2.0/3.0*Dt_SS*ep_bn)*I_10

##Diffusion terms for 4 driven forces respctively
DiffTerm_VV = KT*Diff_VV
DiffTerm_VS = KT*Diff_SV
DiffTerm_SV = KT*Diff_SV
DiffTerm_SS = KT*Diff_SS

CnvcTerm_VV = alpha_V*ep_V.grad.dot(Diff_VV)
CnvcTerm_VS = alpha_S*ep_V.grad.dot(Diff_SV)
CnvcTerm_SV = alpha_V*ep_V.grad.dot(Diff_SV)
CnvcTerm_SS = alpha_S*ep_V.grad.dot(Diff_SS)

#define Transient term:
TranTerm = r**2

'''
*****************
Use a function Diffusion Iteration:
Input the Initial Vacancy concentration, Boundary value
Time step size, and Number of steps
Output Final Vacancy concentration
The concentrations are evaluated in the PolarCoordinate system
*****************
'''

class specie_concentration:
    #define phasefield variables
    def __init__(self):
    #evolution variables
        self.C_V = CellVariable(name = "Vacancy Concentration",mesh = Pmesh, value = 0.0, hasOld = 1)
        self.C_S = CellVariable(name = "Solute Concentration",mesh = Pmesh, value = 0.005, hasOld = 1)
        #plotting variabls
        self.C_V_xy = CellVariable(name = "C_V in XY plane",mesh=Cmesh,value = C_Eq)
        self.C_S_xy = CellVariable(name = "C_S in XY plane",mesh=Cmesh,value = 0.005)
        #psudo variables
        self.CV = CellVariable(name="psudo_vacancy",mesh=Psudomesh,value=0.0)
        self.CS = CellVariable(name="psudo_solute",mesh=Psudomesh,value=0.0)
    #initialize the phase field variables --vacancy concentration and solute concentration
    def initialization(self,vacancy_concentration,solute_concentration):
        self.C_V.setValue(vacancy_concentration)
        self.C_S.setValue(solute_concentration)
    #define time_evolution of species with certain amount of time steps
    def time_evolution(self,delta_t,N_step,inner_vacancy,outer_vacancy,inner_solute,outer_solute):
        self.eqV = TransientTerm(coeff=TranTerm,var=self.C_V) == DiffusionTerm(coeff=DiffTerm_VV,var=self.C_V) + ConvectionTerm(coeff=CnvcTerm_VV,var=self.C_V)\
                                                            + DiffusionTerm(coeff=self.C_V*DiffTerm_VS,var=self.C_S) + ConvectionTerm(coeff=self.C_V*CnvcTerm_VS,var=self.C_S)
        self.eqS = TransientTerm(coeff=TranTerm,var=self.C_S) == DiffusionTerm(coeff=self.C_S*DiffTerm_SV,var=self.C_V) + ConvectionTerm(coeff=self.C_S*CnvcTerm_SV,var=self.C_V)\
                                                            + DiffusionTerm(coeff=self.C_V*DiffTerm_SS,var=self.C_S) + ConvectionTerm(coeff=self.C_V*CnvcTerm_SS,var=self.C_S)
        self.eq = self.eqV & self.eqS ##setup equations
        #define Boundary condition
        self.C_V.constrain(inner_vacancy, Pmesh.facesLeft)
        self.C_V.constrain(outer_vacancy, Pmesh.facesRight)
        #self.C_S.faceGrad.constrain(inner_solute, Pmesh.facesLeft)
        self.C_S.constrain(outer_solute, Pmesh.facesRight)
        #setup iterations:
        timeStepDuration = delta_t
        sum = 0.0
        for n in range(N_step):
            self.eq.solve(dt=timeStepDuration)
            vacancy_adapt = min(timeStepDuration*abs(self.C_V)/abs(self.C_V-self.C_V.old))
            solute_adapt = min(timeStepDuration*abs(self.C_S)/abs(self.C_S-self.C_S.old))
            sum += min(vacancy_adapt,solute_adapt)
            self.C_V.updateOld()
            self.C_S.updateOld()
        return(sum/N_step)
    #define Inverse transformation
    def vacancy_inverse(self,inner_vacancy,outer_vacancy,inner_solute,outer_solute):
        self.CV.setValue(self.C_V)
        self.CS.setValue(self.C_S)
        Xcenter=0.5*np.log((X-CL/2)**2+(Y-CL/2)**2)-np.log(R_in)
        Ycenter=np.arctan2(Y-CL/2,X-CL/2)
        Xleft=np.floor((Xcenter-dLnR/2.0)/dLnR)*dLnR+0.5*dLnR
        Xright=Xleft+dLnR
        Ybottom=np.floor((Ycenter-dCta/2.0)/dCta)*dCta+0.5*dCta
        Ytop=Ybottom+dCta
        UX=(Xright-Xcenter)/dLnR
        UY=(Ytop-Ycenter)/dCta
        self.C_V_xy.setValue((UX*UY*self.CV((Xleft,PA(Ybottom)))+(1-UX)*UY*self.CV((Xright,PA(Ybottom)))+UX*(1-UY)*self.CV((Xleft,PA(Ytop)))+(1-UX)*(1-UY)*self.CV((Xright,PA(Ytop)))))
        self.C_V_xy.setValue(np.nan, where=((X-CL/2)**2+(Y-CL/2)**2<R_in**2))
        self.C_V_xy.setValue(np.nan, where=((X-CL/2)**2+(Y-CL/2)**2>R_out**2))
        self.C_S_xy.setValue((UX*UY*self.CS((Xleft,PA(Ybottom)))+(1-UX)*UY*self.CS((Xright,PA(Ybottom)))+UX*(1-UY)*self.CS((Xleft,PA(Ytop)))+(1-UX)*(1-UY)*self.CS((Xright,PA(Ytop)))))
        self.C_S_xy.setValue(np.nan, where=((X-CL/2)**2+(Y-CL/2)**2<R_in**2))
        self.C_S_xy.setValue(np.nan, where=((X-CL/2)**2+(Y-CL/2)**2>R_out**2))



if __name__ == '__main__':
    OutFile = open('runningTime.txt','w')
    c_Eq = 1/(np.exp(E_f/KT))
    d_t = 5
    N_step = 4
    Ns = 12
    Time = 0.0
    species = specie_concentration()
    species.initialization(C_Eq,C_S_far)
    C_list = [CellVariable(name = "C_V in XY plane",mesh=Cmesh,value = C_Eq) for i in xrange(6)]
    OuterBC = 1/(np.exp((alpha_V*ep_Vf+E_f)/KT))
    InnerBC = 1/(np.exp((alpha_V*ep_Vf+E_f)/KT))
    outer_si = C_S_far
    k = 1
    for i in range(Ns):
        a=time.time()
        Adapt = species.time_evolution(d_t,N_step,InnerBC,OuterBC,0.0,outer_si)
        b=time.time()
        time_duri = b-a
        Time += d_t*N_step
        d_t = 6*d_t
        if(i==2):
            species.vacancy_inverse(20.0,20.0,20.0,20.0)
            C_list[k].setValue(species.C_V_xy)
            k += 1
        if(i==3):
            species.vacancy_inverse(20.0,20.0,20.0,20.0)
            C_list[k].setValue(species.C_V_xy)
            k += 1
        if(i==4):
            species.vacancy_inverse(20.0,20.0,20.0,20.0)
            C_list[k].setValue(species.C_V_xy)
            k += 1
        print d_t,time_duri
    species.vacancy_inverse(20.0,20.0,20.0,20.0)
    C_list[4].setValue(species.C_V_xy)
    C_list[5].setValue(species.C_V_xy)
    Ind_list = [" "," "," "," "," "," "]
    PL_con = PF.vac_conc_2Dcontour(100.,2000)
    PL_con.get_color_map(C_Eq*20.,0.,"jet")
    #PL_con.making_contour(Plot_single,6.7,50.)
    PL_con.making_multi_hori(C_list,6.7,50.,Ind_list,17,12,color_bar = True)
    '''
    spe = specie_concentration()
    spe.initialization(species.C_V,species.C_S)
    spe.vacancy_inverse(20.0,20.0,20.0,20.0)
    viewer = Matplotlib2DGridContourViewer(vars = species.C_V_xy*1E6,
                              datamin = C_Eq*0.0 ,
                              datamax = C_Eq*1E7)
    viewer.plot()
    OuterBC = 1/(np.exp((alpha_V*ep_Vf+E_f)/KT))
    InnerBC = 1/(np.exp((alpha_V*ep_Vf+E_f)/KT))
    outer_si = C_S_far
    for i in range(Ns):
        a=time.time()
        Adapt = spe.time_evolution(d_t,N_step,InnerBC,OuterBC,0.0,outer_si)
        b=time.time()
        time_duri = b-a
        Time += d_t*N_step
        d_t = 1.7*d_t#min(2.0*d_t,0.15*Adapt)
        spe.vacancy_inverse(20.0,20.0,20.0,20.0)
        if(i%2==0):
            #viewer = Matplotlib2DGridContourViewer(vars = species.C_V_xy*1E6, datamin = C_Eq*0.0, datamax = C_Eq*5E6)
            viewer.plot()
        print d_t,time_duri
    viewer._promptForOpinion()
    '''




######Plotting maximum value has been modified
######Outer B.C.





