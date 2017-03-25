__author__ = 'zeboli1'

import numpy as np
from scipy import linalg as LA


class finite_diff_evo:
    #define finite_difference evolution scheme
    def __init__(self,var_initial):
        self.var = np.array(var_initial)
        self.dimension = len(var_initial)

    #time evolution step
    def evol_step(self,trans_matx,delta_t,source = []):
        n_dim = self.dimension
        if(source == []):
            source = np.zeros((n_dim))
        d_var = delta_t*(np.dot(trans_matx,self.var) + source)
        var_temp = self.var + d_var
        d_var = delta_t*(np.dot(trans_matx,var_temp) + source)
        self.var = self.var + d_var
        return d_var/delta_t
    #adaptive time step calculation
    def adaptive_step(self,kappa,deriv):
        return kappa*min(abs(self.var/deriv))

class eigen_decom_evo:
    #define the eigen_value_decomp evolution scheme
    def __init__(self,var_initial):
        self.var = np.array(var_initial)
        self.var_temp = np.array(var_initial)
        self.dimension = len(var_initial)

    #time evolution scheme
    def get_eigen_info(self,matrix,n_dim = -1):
        if(n_dim < 0):
            n_dim = self.dimension
        val,left_vec,right_vec = LA.eig(matrix,left=True, right=True)
        #sort eigenvalues:
        idx = val.argsort()[::-1]
        eigen_value = val[idx]
        left_eigen_vector = left_vec[:,idx]
        right_eigen_vector = right_vec[:,idx]
        #Normalization: change the magnitudes of left eigenvectors
        Gamma = np.dot(left_eigen_vector.transpose(),right_eigen_vector)
        elements = []
        for i in range(n_dim):
            elements.append(1./Gamma[i][i])
        Diag = np.diag(elements)
        left_eigen_vector = np.dot(left_eigen_vector,Diag)
        return np.real(eigen_value),np.real(right_eigen_vector),np.real(left_eigen_vector)


    #time evolution step
    def evol_step(self,trans_matx,delta_t,var,source = [],min = 1.E-20):
        n_dim = self.dimension
        if(source == []):
            source = np.zeros((n_dim))
        egval,right_egvec,left_egvec = self.get_eigen_info(trans_matx)
        coeff = np.dot(left_egvec.transpose(),var)
        src = np.dot(left_egvec.transpose(),source)
        coeff = coeff*np.exp(delta_t*egval)
        for i in range(n_dim):
            if(abs(egval[i]) < min):
                src[i] = delta_t*src[i]
            else:
                src[i] = src[i]/egval[i]*(np.exp(delta_t*egval[i]) - 1)
        coeff = coeff + src
        self.var_temp = np.dot(right_egvec,coeff)
        return (self.var_temp - var)/delta_t

    #compute the adaptive steps
    def adaptive_step(self,kappa,deriv):
        return kappa*min(abs(self.var/deriv))

    #update variables
    def update(self):
        self.var = self.var_temp