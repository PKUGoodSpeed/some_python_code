__author__ = 'zeboli1'

import numpy as np


def judge(list,element,error):
    n = len(list)
    for i in range(n):
        if(np.dot(np.array(list[i])- np.array(element),np.array(list[i])- np.array(element)) < error):
            return 1
    return 0


'''
lattice structure
'''
class latt_struct:
    #define different types of lattice structures

    def __init__(self,type = 1):
        self.basis = []
        self.threeD_neighbor = []
        self.num_basis = 0
        self.num_neibr = 0
        self.num_twoDnb = 0
        self.twoD_neighbor = []
        if(type == 0):
            #cubic
            self.num_basis = 1
            self.basis.append([0., 0., 0.])
            self.num_neibr = 6
            self.threeD_neighbor.append([1., 0., 0.])
            self.threeD_neighbor.append([-1., 0., 0.])
            self.threeD_neighbor.append([0., 1., 0.])
            self.threeD_neighbor.append([0., -1., 0.])
            self.threeD_neighbor.append([0., 0., 1.])
            self.threeD_neighbor.append([0., 0., -1.])
        elif(type == 1):
            #fcc
            self.num_basis = 4
            self.basis.append([0., 0., 0.])
            self.basis.append([0.5, 0.5, 0.])
            self.basis.append([0.5, 0., 0.5])
            self.basis.append([0., 0.5, 0.5])
            self.num_neibr = 12
            self.threeD_neighbor.append([0.5, 0.5, 0.0])
            self.threeD_neighbor.append([0.5, -0.5, 0.0])
            self.threeD_neighbor.append([-0.5, 0.5, 0.0])
            self.threeD_neighbor.append([-0.5, -0.5, 0.0])
            self.threeD_neighbor.append([0.0, 0.5, 0.5])
            self.threeD_neighbor.append([0.0, 0.5, -0.5])
            self.threeD_neighbor.append([0.0, -0.5, 0.5])
            self.threeD_neighbor.append([0.0, -0.5, -0.5])
            self.threeD_neighbor.append([0.5, 0.0, 0.5])
            self.threeD_neighbor.append([0.5, 0.0, -0.5])
            self.threeD_neighbor.append([-0.5, 0.0, 0.5])
            self.threeD_neighbor.append([-0.5, 0.0, -0.5])
        elif(type == 2):
            #bcc
            self.num_basis = 2
            self.basis.append([0., 0., 0.])
            self.basis.append([0.5, 0.5, 0.5])
            self.num_neibr = 8
            self.threeD_neighbor.append([0.5, 0.5, 0.5])
            self.threeD_neighbor.append([0.5, 0.5, -0.5])
            self.threeD_neighbor.append([0.5, -0.5, 0.5])
            self.threeD_neighbor.append([0.5, -0.5, -0.5])
            self.threeD_neighbor.append([-0.5, 0.5, 0.5])
            self.threeD_neighbor.append([-0.5, 0.5, -0.5])
            self.threeD_neighbor.append([-0.5, -0.5, 0.5])
            self.threeD_neighbor.append([-0.5, -0.5, -0.5])
        else:
            #excluding other cases
            self.num_basis = 0
            self.num_neibr = 0

    def create_twoD_neighbors(self,x_axis,y_axis,error = 1.E-6):
        #create 2D projected neighbor coordinates
        for i in range(self.num_neibr):
            x = np.dot(np.array(self.threeD_neighbor[i]),np.array(x_axis))
            y = np.dot(np.array(self.threeD_neighbor[i]),np.array(y_axis))
            new = np.array([x,y])
            if(judge(self.twoD_neighbor,new,error) == 0 and np.dot(new,new)>error):
                self.twoD_neighbor.append(new)
        self.num_twoDnb = len(self.twoD_neighbor)


'''
For nickel lattice
'''
a = 3.52 #A : lattice constant for nickel
R_in = 6.7 #A : radius for discretized region
R_buff = 9.6 #A : radius for buffer region
nu_Poisson = 0.31 # : Poisson ratio for nickle
burg = a/np.sqrt(2.) #A :Burger's vector

'''
For edge dislocation
'''
b_vec = [1./np.sqrt(2.),-1./np.sqrt(2.),0.] #Burger's vector
n_vec = [1./np.sqrt(3.),1./np.sqrt(3.),1./np.sqrt(3.)] #Slip plane normal
t_vec = [1./np.sqrt(6.),1./np.sqrt(6.),-2./np.sqrt(6.)] #Threading direction


'''
Vacancy formation energy fit
'''
V_szmisfit = 18.1#eV vacancy size-misfit
Si_szmisfit = -0.251#eV Si size-misfit
E_f = 1.62940680#eV vacancy formation energy for nickel
b_coeff = 6.37383436#eV Fitting parameter with strain
c_coeff = -17.07913247#eV Fitting parameter with strain
